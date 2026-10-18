# -*- coding: utf-8 -*-
"""Data normalization, preprocessing and time-series helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ALIASES = {
    "date": ("date", "ngay", "ngày", "order_date", "transaction_date"),
    "store_nbr": ("store_nbr", "store", "store_id", "ma_cua_hang"),
    "family": ("family", "product", "san_pham", "sản phẩm", "sku", "item"),
    "sales": ("sales", "quantity", "qty", "doanh_so", "so_luong", "số lượng"),
    "onpromotion": ("onpromotion", "promotion", "promotions", "khuyen_mai"),
}


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    rename = {}
    for target, aliases in ALIASES.items():
        source = next((name for name in aliases if name in result.columns), None)
        if source is not None and source != target and target not in result.columns:
            rename[source] = target
    return result.rename(columns=rename)


def load_sales_data(file, date_col="date", product_col="product", qty_col="quantity"):
    """Read a generic CSV/Excel sales file and normalize it to the app schema."""
    if hasattr(file, "name") and file.name.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(file)
    else:
        frame = pd.read_csv(file)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    explicit = {
        date_col.lower(): "date",
        product_col.lower(): "family",
        qty_col.lower(): "sales",
    }
    frame = frame.rename(
        columns={key: value for key, value in explicit.items() if key in frame.columns}
    )
    frame = _normalize_columns(frame)
    if "date" not in frame or "sales" not in frame:
        raise ValueError(
            "File cần có cột ngày và doanh số/số lượng (date, sales/quantity)."
        )
    if "family" not in frame:
        frame["family"] = "Tổng hợp"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    if "onpromotion" in frame:
        frame["onpromotion"] = pd.to_numeric(frame["onpromotion"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Không tìm thấy ngày hợp lệ trong dữ liệu.")
    return frame


def clean_sales_data(
    data: pd.DataFrame,
    *,
    fill_missing_sales: bool = True,
    remove_duplicates: bool = True,
    cap_outliers: bool = False,
) -> pd.DataFrame:
    """Apply explicit, repeatable cleaning choices without mutating source data."""
    cleaned = _normalize_columns(data)
    if "date" not in cleaned or "sales" not in cleaned:
        raise ValueError("Dữ liệu phải có cột date và sales.")

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["sales"] = pd.to_numeric(cleaned["sales"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])
    if fill_missing_sales:
        cleaned["sales"] = cleaned["sales"].fillna(0)
    else:
        cleaned = cleaned.dropna(subset=["sales"])

    if cleaned.empty:
        raise ValueError(
            "Không còn dòng hợp lệ sau khi xử lý ngày và doanh số; "
            "hãy kiểm tra dữ liệu hoặc tùy chọn thay giá trị thiếu."
        )

    for column in ("family", "store_nbr"):
        if column in cleaned:
            cleaned[column] = cleaned[column].fillna("Không xác định")

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates(keep="first")
        if "id" in cleaned.columns:
            repeated_ids = cleaned["id"].notna() & cleaned.duplicated(
                subset=["id"], keep=False
            )
            if repeated_ids.any():
                duplicate_ids = (
                    cleaned.loc[repeated_ids, "id"].drop_duplicates().head(5).tolist()
                )
                raise ValueError(
                    "Dữ liệu có các bản ghi trùng id nhưng nội dung khác nhau "
                    f"(ví dụ: {duplicate_ids}). Hãy kiểm tra dữ liệu trước khi tiếp tục."
                )

    if cap_outliers and not cleaned.empty:
        lower = cleaned["sales"].quantile(0.01)
        upper = cleaned["sales"].quantile(0.99)
        cleaned["sales"] = cleaned["sales"].clip(lower=lower, upper=upper)

    return cleaned.sort_values("date").reset_index(drop=True)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, promotion, lag and rolling features without target leakage."""
    frame = _normalize_columns(data)
    if "date" not in frame or "sales" not in frame:
        raise ValueError("Cần có cột date và sales để tạo đặc trưng.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["date", "sales"]).sort_values("date").reset_index(drop=True)

    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype("int8")
    if "onpromotion" not in frame:
        frame["onpromotion"] = 0
    frame["onpromotion"] = pd.to_numeric(
        frame["onpromotion"], errors="coerce"
    ).fillna(0)

    group_columns = [c for c in ("store_nbr", "family") if c in frame.columns]
    if group_columns:
        groups = frame.groupby(
            group_columns, dropna=False, sort=False, observed=True
        ).indices.values()
    else:
        groups = (np.arange(len(frame)),)

    sales = frame["sales"].to_numpy(dtype=np.float64, na_value=np.nan)
    lag_1 = np.full(len(frame), np.nan)
    lag_7 = np.full(len(frame), np.nan)
    lag_28 = np.full(len(frame), np.nan)
    rolling_mean_7 = np.full(len(frame), np.nan)
    rolling_mean_28 = np.full(len(frame), np.nan)

    for positions in groups:
        values = sales[positions]
        size = len(values)
        for lag, result in ((1, lag_1), (7, lag_7), (28, lag_28)):
            if size > lag:
                result[positions[lag:]] = values[:-lag]

        finite = np.isfinite(values)
        sums = np.empty(size + 1, dtype=np.float64)
        sums[0] = 0
        np.cumsum(np.where(finite, values, 0), out=sums[1:])
        counts = np.empty(size + 1, dtype=np.int64)
        counts[0] = 0
        np.cumsum(finite, dtype=np.int64, out=counts[1:])
        row_numbers = np.arange(size)

        for window, result in ((7, rolling_mean_7), (28, rolling_mean_28)):
            starts = np.maximum(0, row_numbers - window)
            window_counts = counts[row_numbers] - counts[starts]
            means = np.full(size, np.nan)
            np.divide(
                sums[row_numbers] - sums[starts],
                window_counts,
                out=means,
                where=window_counts > 0,
            )
            result[positions] = means

    frame["lag_1"] = lag_1
    frame["lag_7"] = lag_7
    frame["lag_28"] = lag_28
    frame["rolling_mean_7"] = rolling_mean_7
    frame["rolling_mean_28"] = rolling_mean_28

    return frame


def aggregate_daily(
    data: pd.DataFrame,
    product: str | None = None,
    freq: str = "D",
    store: str | None = None,
    family: str | None = None,
) -> pd.DataFrame:
    """Aggregate sales over time, optionally for one store and product family."""
    frame = _normalize_columns(data)
    if product and product != "Tất cả nhóm hàng" and "family" in frame:
        frame = frame[frame["family"].astype(str) == str(product)]
    if family and family != "Tất cả nhóm hàng" and "family" in frame:
        frame = frame[frame["family"].astype(str) == str(family)]
    if store and store != "Tất cả cửa hàng" and "store_nbr" in frame:
        frame = frame[frame["store_nbr"].astype(str) == str(store)]
    frame = frame.dropna(subset=["date"]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return pd.DataFrame(columns=["date", "sales"])
    return (
        frame.set_index("date")
        .resample(freq)["sales"]
        .sum()
        .rename("sales")
        .reset_index()
    )


def create_sequences(series, look_back=14, horizon=1):
    """Create sliding windows for a univariate recurrent neural network."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if look_back < 1 or horizon < 1:
        raise ValueError("look_back và horizon phải lớn hơn 0.")
    X, y = [], []
    for index in range(len(values) - look_back - horizon + 1):
        X.append(values[index : index + look_back])
        y.append(values[index + look_back : index + look_back + horizon])
    return np.asarray(X), np.asarray(y)


def scale_series(series):
    """Fit a MinMax scaler and return flattened scaled values and the scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = np.asarray(series, dtype=float).reshape(-1, 1)
    if values.size == 0:
        raise ValueError("Không thể chuẩn hóa chuỗi rỗng.")
    return scaler.fit_transform(values).flatten(), scaler


def train_test_split_series(series, test_ratio=0.2):
    """Split time series chronologically; never shuffle future observations."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio phải nằm trong khoảng (0, 1).")
    values = np.asarray(series)
    split_index = int(len(values) * (1 - test_ratio))
    return values[:split_index], values[split_index:]


def compute_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE, and WAPE with zero-demand-safe denominators."""
    actual = np.asarray(y_true, dtype=float).flatten()
    predicted = np.asarray(y_pred, dtype=float).flatten()
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true và y_pred phải có cùng độ dài và không được rỗng.")
    absolute_error = np.abs(actual - predicted)
    nonzero = actual != 0
    mape = (
        float(np.mean(absolute_error[nonzero] / np.abs(actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    denominator = float(np.abs(actual).sum())
    wape = float(absolute_error.sum() / denominator * 100) if denominator else float("nan")
    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAPE (%)": mape,
        "WAPE (%)": wape,
    }
