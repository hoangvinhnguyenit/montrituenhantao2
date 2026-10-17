# -*- coding: utf-8 -*-
"""Local-file and optional Supabase adapters for the ERD data model."""

from __future__ import annotations

import io
import os
from queue import Empty, Queue
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from utils import load_sales_data


ERD_TABLES = (
    "train",
    "test",
    "stores",
    "transactions",
    "oil",
    "holidays_events",
    "sample_submission",
)

SUPABASE_SELECT_COLUMNS = {
    "train": "id,date,store_nbr,family,sales,onpromotion",
    "test": "id,date,store_nbr,family,onpromotion",
    "stores": "store_nbr,city,state,type,cluster",
    "transactions": "date,store_nbr,transactions",
    "oil": "date,dcoilwtico",
    "holidays_events": "date,type,locale,locale_name,description",
    "sample_submission": "id,sales",
}

TABLE_ALIASES = {
    "train": "train",
    "train2": "train",
    "test": "test",
    "stores": "stores",
    "transactions": "transactions",
    "oil": "oil",
    "holidays": "holidays_events",
    "holidays_events": "holidays_events",
    "sample_submission": "sample_submission",
}


def _table_name(filename: str) -> str | None:
    stem = Path(filename).stem.strip().lower()
    return TABLE_ALIASES.get(stem)


def _read_csv_bytes(name: str, payload: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Không đọc được file CSV {name}: {exc}") from exc


def _read_upload(upload) -> dict[str, pd.DataFrame]:
    filename = upload.name.lower()
    if filename.endswith(".zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(upload.getvalue()))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{upload.name} không phải file ZIP hợp lệ.") from exc
        tables = {}
        for member in archive.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            table = _table_name(member)
            if table:
                tables[table] = _read_csv_bytes(
                    member, archive.read(member)
                )
        return tables

    table = _table_name(upload.name)
    if filename.endswith(".csv"):
        data = _read_csv_bytes(upload.name, upload.getvalue())
    elif filename.endswith((".xlsx", ".xls")):
        data = pd.read_excel(upload)
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {upload.name}")
    if table is None:
        table = "generic_sales"
    return {table: data}


def load_uploaded_sources(uploads) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Read individual CSVs, Excel, or a Kaggle-style ZIP and join by ERD keys."""
    tables = {}
    for upload in uploads:
        for name, frame in _read_upload(upload).items():
            if name in tables:
                raise ValueError(f"Bạn đã tải lên nhiều hơn một file cho bảng {name}.")
            tables[name] = frame

    if not tables:
        raise ValueError("Chưa có file dữ liệu để nạp.")
    if "train" not in tables:
        generic = tables.get("generic_sales")
        if generic is None and len(tables) == 1:
            generic = next(iter(tables.values()))
        if generic is None:
            raise ValueError("Cần file train.csv làm bảng doanh số.")
        sales = load_sales_data(io.BytesIO(generic.to_csv(index=False).encode("utf-8")))
        return sales, None, tables
    sales, test = normalize_erd_tables(tables)
    return sales, test, tables


def _normalized_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def _assert_unique(frame: pd.DataFrame, keys: list[str], table_name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(
            f"Bảng {table_name} vi phạm khóa duy nhất theo "
            f"{', '.join(keys)}. Hãy xử lý bản ghi trùng trước khi nối."
        )


def _deduplicate_identical_keys(
    frame: pd.DataFrame, key: str, table_name: str
) -> pd.DataFrame:
    repeated = frame.duplicated(key, keep=False)
    if not repeated.any():
        return frame

    deduplicated = frame.drop_duplicates(keep="first")
    if deduplicated.duplicated(key).any():
        duplicate_keys = (
            deduplicated.loc[deduplicated.duplicated(key, keep=False), key]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Bảng {table_name} có các bản ghi trùng {key} nhưng nội dung "
            f"khác nhau (ví dụ: {duplicate_keys}). Hãy sửa dữ liệu trước khi nối."
        )
    return deduplicated


def normalize_erd_tables(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Join the ERD tables without multiplying training/test fact rows."""
    if "train" not in tables:
        raise ValueError("Thiếu bảng train (bảng doanh số huấn luyện).")

    stores = _normalized_table(tables["stores"]) if "stores" in tables else None
    transactions = (
        _normalized_table(tables["transactions"])
        if "transactions" in tables
        else None
    )
    oil = _normalized_table(tables["oil"]) if "oil" in tables else None
    holidays = (
        _normalized_table(tables["holidays_events"])
        if "holidays_events" in tables
        else None
    )

    if stores is not None:
        if "store_nbr" not in stores:
            raise ValueError("Bảng stores cần có khóa store_nbr.")
        _assert_unique(stores, ["store_nbr"], "stores")

    if transactions is not None:
        required = {"date", "store_nbr", "transactions"}
        if not required.issubset(transactions.columns):
            raise ValueError(
                "Bảng transactions cần có date, store_nbr và transactions."
            )
        # This is the ERD's composite key. Sum repeated imports before joining.
        transactions = (
            transactions.groupby(["date", "store_nbr"], as_index=False, dropna=False)[
                "transactions"
            ]
            .sum(min_count=1)
        )

    if oil is not None and "date" in oil:
        if "dcoilwtico" not in oil:
            candidates = [column for column in oil if column != "date"]
            if candidates:
                oil = oil.rename(columns={candidates[0]: "dcoilwtico"})
        if "dcoilwtico" in oil:
            oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
            oil = oil.groupby("date", as_index=False)["dcoilwtico"].mean()

    holiday_by_date = None
    if holidays is not None and "date" in holidays:
        holiday_columns = [
            column
            for column in ("type", "locale", "locale_name", "description")
            if column in holidays
        ]
        holiday_by_date = (
            holidays.groupby("date", dropna=False)
            .agg(
                has_holiday=("date", "size"),
                holiday_types=(
                    holiday_columns[0],
                    lambda values: ", ".join(
                        sorted(values.dropna().astype(str).unique())
                    ),
                )
                if holiday_columns
                else ("date", lambda values: "Ngày lễ"),
            )
            .reset_index()
        )

    def enrich(facts: pd.DataFrame, *, has_target: bool) -> pd.DataFrame:
        facts = _normalized_table(facts)
        if not {"date", "store_nbr", "family"}.issubset(facts.columns):
            raise ValueError(
                "Bảng train/test cần có date, store_nbr và family theo ERD."
            )
        if "id" not in facts:
            raise ValueError("Bảng train/test cần có khóa chính id theo ERD.")
        facts = _deduplicate_identical_keys(
            facts, "id", "train" if has_target else "test"
        )
        if has_target and "sales" not in facts:
            raise ValueError("Bảng train cần có cột sales.")
        if "onpromotion" not in facts:
            facts["onpromotion"] = 0
        facts["onpromotion"] = pd.to_numeric(
            facts["onpromotion"], errors="coerce"
        ).fillna(0)
        facts["store_nbr"] = facts["store_nbr"].astype("string")

        if stores is not None:
            dimension = stores.copy()
            dimension["store_nbr"] = dimension["store_nbr"].astype("string")
            facts = facts.merge(
                dimension, on="store_nbr", how="left", validate="many_to_one"
            )
        if transactions is not None:
            transaction_dim = transactions.copy()
            transaction_dim["store_nbr"] = transaction_dim["store_nbr"].astype(
                "string"
            )
            facts = facts.merge(
                transaction_dim,
                on=["date", "store_nbr"],
                how="left",
                validate="many_to_one",
            )
        if oil is not None and "dcoilwtico" in oil:
            facts = facts.merge(oil, on="date", how="left", validate="many_to_one")
        if holiday_by_date is not None:
            facts = facts.merge(
                holiday_by_date, on="date", how="left", validate="many_to_one"
            )
            facts["has_holiday"] = facts["has_holiday"].fillna(0).astype("int16")
        facts["family"] = facts["family"].astype("string").fillna("Không xác định")
        if has_target:
            facts["sales"] = pd.to_numeric(facts["sales"], errors="coerce")
        return facts.sort_values("date").reset_index(drop=True)

    sales = enrich(tables["train"], has_target=True)
    test = enrich(tables["test"], has_target=False) if "test" in tables else None
    return sales, test


def get_supabase_configuration() -> tuple[str | None, str | None]:
    """Read server-side credentials from environment variables or Streamlit secrets."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    try:
        if not url:
            url = st.secrets.get("SUPABASE_URL")
        if not key:
            key = st.secrets.get("SUPABASE_KEY")
    except StreamlitSecretNotFoundError:
        pass
    return url, key


def load_supabase_sources(
    *,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Load accessible ERD tables concurrently in bounded API pages."""
    url, key = get_supabase_configuration()
    if not url or not key:
        raise ValueError("Hãy cấu hình SUPABASE_URL và SUPABASE_KEY ở môi trường máy chủ.")
    try:
        import httpx
        from supabase import create_client
        from postgrest.exceptions import APIError
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu gói supabase-py. Cài bằng `pip install supabase` để bật nguồn này."
        ) from exc

    tables = {}
    progress_events = Queue()

    def read_table(table: str) -> list[dict]:
        for attempt in range(2):
            client = create_client(url, key)
            try:
                rows = []
                offset = 0
                page_size = 10000
                while True:
                    columns = SUPABASE_SELECT_COLUMNS.get(
                        "train" if table == "train2" else table, "*"
                    )
                    query = client.table(table).select(columns).range(
                        offset, offset + page_size - 1
                    )
                    try:
                        result = query.execute()
                    except APIError as exc:
                        if (
                            columns == "*"
                            or getattr(exc, "code", None) not in {"42703", "PGRST204"}
                        ):
                            raise
                        result = (
                            client.table(table)
                            .select("*")
                            .range(offset, offset + page_size - 1)
                            .execute()
                        )
                    page = result.data or []
                    rows.extend(page)
                    if progress_callback is not None:
                        progress_events.put((table, len(rows)))
                    if not page:
                        break
                    offset += len(page)
                return rows
            except httpx.TransportError:
                if attempt == 1:
                    raise
            finally:
                client.postgrest.session.close()
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Không thể đọc bảng {table} sau khi thử kết nối lại.")

    def is_missing_table(error: APIError) -> bool:
        return getattr(error, "code", None) in {"PGRST205", "42P01"}

    table_rows = {}
    table_errors = {}
    transport_errors = {}

    def report_progress() -> None:
        latest = {}
        while True:
            try:
                table, row_count = progress_events.get_nowait()
            except Empty:
                break
            latest[table] = row_count
        if progress_callback is not None:
            for table, row_count in latest.items():
                progress_callback(table, row_count)

    with ThreadPoolExecutor(max_workers=min(4, len(ERD_TABLES))) as executor:
        pending = {
            executor.submit(read_table, table): table for table in ERD_TABLES
        }
        while pending:
            completed, _ = wait(
                pending, timeout=0.5, return_when=FIRST_COMPLETED
            )
            report_progress()
            for future in completed:
                table = pending.pop(future)
                try:
                    rows = future.result()
                except APIError as exc:
                    table_errors[table] = exc
                    continue
                except httpx.TransportError as exc:
                    transport_errors[table] = exc
                    continue
                table_rows[table] = rows
                if progress_callback is not None:
                    progress_callback(table, len(rows))

    for table in ERD_TABLES:
        transport_error = transport_errors.get(table)
        if transport_error is not None:
            raise RuntimeError(
                f"Kết nối tới Supabase bị ngắt khi đọc bảng {table}, "
                "kể cả sau khi đã tự thử lại. Hãy thử nạp lại sau ít phút."
            ) from transport_error

        error = table_errors.get(table)
        if error is not None:
            if not is_missing_table(error):
                code = getattr(error, "code", None) or "không xác định"
                raise RuntimeError(
                    f"Không thể đọc bảng {table} từ Supabase (mã lỗi {code}). "
                    "Kiểm tra API key, quyền SELECT/RLS và cấu trúc bảng."
                ) from error
            if table != "train":
                continue
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(read_table, "train2")
                    while not future.done():
                        wait((future,), timeout=0.5)
                        report_progress()
                    rows = future.result()
            except httpx.TransportError as transport_error:
                raise RuntimeError(
                    "Kết nối tới Supabase bị ngắt khi đọc bảng train2, "
                    "kể cả sau khi đã tự thử lại. Hãy thử nạp lại sau ít phút."
                ) from transport_error
            except APIError as legacy_error:
                if is_missing_table(legacy_error):
                    raise ValueError(
                        "Không tìm thấy bảng train hoặc train2 trong Supabase. "
                        "Tạo bảng train hoặc đổi tên train2 thành train."
                    ) from legacy_error
                code = getattr(legacy_error, "code", None) or "không xác định"
                raise RuntimeError(
                    f"Không thể đọc bảng train2 từ Supabase (mã lỗi {code}). "
                    "Kiểm tra API key, quyền SELECT/RLS và cấu trúc bảng."
                ) from legacy_error
            if progress_callback is not None:
                progress_callback("train2", len(rows))
            if rows:
                tables["train"] = pd.DataFrame(rows)
            continue

        rows = table_rows[table]
        if rows:
            tables[table] = pd.DataFrame(rows)

    if "train" not in tables:
        raise ValueError("Không tìm thấy bảng train hoặc bảng này không có dữ liệu.")
    if progress_callback is not None:
        progress_callback("Chuẩn hóa dữ liệu", sum(len(frame) for frame in tables.values()))
    sales, test = normalize_erd_tables(tables)
    return sales, test, tables


def make_sample_dataset(days: int = 730) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a compact synthetic dataset that follows the ERD relationships."""
    random = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    stores = pd.DataFrame(
        [
            {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
            {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 2},
            {"store_nbr": 3, "city": "Cuenca", "state": "Azuay", "type": "A", "cluster": 1},
        ]
    )
    families = ["GROCERY I", "BEVERAGES", "PRODUCE", "DAIRY"]
    facts = []
    transactions = []
    for store in stores["store_nbr"]:
        store_multiplier = 0.8 + 0.25 * int(store)
        for family_index, family in enumerate(families):
            base = 32 + family_index * 13
            weekly = 8 * np.sin(2 * np.pi * dates.dayofweek / 7)
            seasonal = 6 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            promotion = random.binomial(1, 0.12, size=days)
            noise = random.normal(0, 5, size=days)
            sales = np.maximum(
                0, store_multiplier * (base + weekly + seasonal + 18 * promotion + noise)
            )
            for index, date in enumerate(dates):
                facts.append(
                    {
                        "id": len(facts) + 1,
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": round(float(sales[index]), 2),
                        "onpromotion": int(promotion[index]),
                    }
                )
        transactions.extend(
            {
                "date": date,
                "store_nbr": store,
                "transactions": int(random.integers(250, 900)),
            }
            for date in dates
        )

    oil = pd.DataFrame(
        {"date": dates, "dcoilwtico": 65 + 4 * np.sin(np.arange(days) / 45)}
    )
    holiday_dates = dates[dates.dayofweek == 6][::7]
    holidays = pd.DataFrame(
        {
            "date": holiday_dates,
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Ngày lễ mẫu",
            "transferred": False,
        }
    )
    tables = {
        "train": pd.DataFrame(facts),
        "stores": stores,
        "transactions": pd.DataFrame(transactions),
        "oil": oil,
        "holidays_events": holidays,
    }
    sales, _ = normalize_erd_tables(tables)
    return sales, tables
