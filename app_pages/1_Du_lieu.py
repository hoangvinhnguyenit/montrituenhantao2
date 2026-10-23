# -*- coding: utf-8 -*-
"""Load local Kaggle-style files or connect to Supabase."""

import pandas as pd
import streamlit as st

from app_common import configure_page, initialize_session_state
from data_repository import (
    ERD_TABLES,
    get_supabase_configuration,
    load_supabase_sources,
    load_uploaded_sources,
    make_sample_dataset,
)


configure_page("Dữ liệu")
initialize_session_state()

st.title("Dữ liệu bán hàng")
st.caption(
    "Nạp dữ liệu theo 2 cách: nạp trực tiếp file .csv (nhanh nhất)"
    "hoặc đồng bộ dữ liệu từ Supabase (thời gian chờ lâu hơn)"
)


def save_dataset(sales, test, tables, source):
    st.session_state.raw_df = sales
    st.session_state.processed_df = None
    st.session_state.test_df = test
    st.session_state.source_tables = tables
    st.session_state.source_label = source
    for key, default in (
        ("model", None),
        ("scaler", None),
        ("test_metrics", None),
        ("test_pred_data", None),
        ("training_history", None),
        ("forecast_result", None),
        ("submission_result", None),
        ("inventory_plan", None),
        ("training_filter", None),
        ("training_series", None),
        ("full_scaled", None),
        ("model_config", None),
    ):
        st.session_state[key] = default


local_tab, supabase_tab = st.tabs(["Tệp cục bộ", "Đồng bộ dữ liệu (Supabase)"])
with local_tab:
    files = st.file_uploader(
        "Chọn CSV/Excel hoặc ZIP chứa các bảng ERD",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
        help=(
            "Tên file được nhận diện theo train, test, stores, "
            "transactions, oil, holidays_events, sample_submission."
        ),
    )
    load_button = st.button("Nạp dữ liệu", type="primary", disabled=not files)
    sample_button = st.button("Dùng dữ liệu mẫu theo ERD")
    if load_button:
        try:
            sales, test, tables = load_uploaded_sources(files)
            save_dataset(sales, test, tables, "Tệp cục bộ")
            st.success(f"Đã nạp {len(sales):,} dòng bán hàng.")
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Không thể nạp dữ liệu: {exc}")
    if sample_button:
        sales, tables = make_sample_dataset()
        save_dataset(sales, None, tables, "Dữ liệu mẫu")
        st.success(f"Đã tạo dữ liệu mẫu theo ERD ({len(sales):,} dòng).")

    st.markdown("**Khóa nối:** train/test → stores theo `store_nbr`; transactions theo "
                "`date + store_nbr`; oil và holidays_events theo `date`.")
    st.caption(
        "Có thể tải trực tiếp file train.csv và stores/transactions/oil/holidays_events "
        "từ bộ dữ liệu Favorita. File có tên khác sẽ được đọc như bảng doanh số tổng quát."
    )

with supabase_tab:
    supabase_url, supabase_key = get_supabase_configuration()
    if supabase_url and supabase_key:
        st.success("Đã tìm thấy cấu hình Supabase phía máy chủ; khóa không được hiển thị.")
        st.caption(
            "Ứng dụng nạp song song tối đa 4 bảng, mỗi lần yêu cầu tối đa 1.000 dòng/1s "
            "và tự thử lại một lần nếu kết nối bị ngắt."
        )
        if st.button("Nạp từ Supabase", type="primary"):
            load_status = st.status("Đang kết nối Supabase...", expanded=True)

            def show_load_progress(table, row_count):
                if table == "Chuẩn hóa dữ liệu":
                    label = "Đang ghép và chuẩn hóa dữ liệu Supabase..."
                else:
                    label = f"Đang tải bảng {table}: {row_count:,} dòng"
                load_status.update(
                    label=label,
                    state="running",
                )

            try:
                sales, test, tables = load_supabase_sources(
                    progress_callback=show_load_progress
                )
                save_dataset(sales, test, tables, "Supabase")
                load_status.update(
                    label="Đã nạp xong toàn bộ dữ liệu Supabase.",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    f"Đã nạp {len(sales):,} dòng doanh số và "
                    f"{len(tables):,} bảng từ Supabase."
                )
            except (ValueError, RuntimeError) as exc:
                load_status.update(label="Nạp dữ liệu thất bại.", state="error")
                st.error(f"Không thể đọc Supabase: {exc}")
    else:
        st.info(
            "Khi cần kết nối, đặt `SUPABASE_URL` và `SUPABASE_KEY` trong biến môi "
            "trường của máy chủ Streamlit. Không đưa service-role key vào mã nguồn."
        )
    st.code(
        "\n".join(f"{table}" for table in ERD_TABLES),
        language="text",
    )

sales = st.session_state.raw_df
if sales is not None and not sales.empty:
    st.divider()
    st.subheader(f"Đang dùng: {st.session_state.source_label or 'dữ liệu đã nạp'}")
    first, second, third, fourth = st.columns(4)
    first.metric("Số dòng", f"{len(sales):,}")
    second.metric("Nhóm hàng", f"{sales['family'].nunique():,}")
    third.metric(
        "Cửa hàng",
        f"{sales['store_nbr'].nunique():,}" if "store_nbr" in sales else "—",
    )
    fourth.metric(
        "Khoảng ngày",
        f"{sales['date'].min():%d/%m/%Y} – {sales['date'].max():%d/%m/%Y}",
    )
    st.dataframe(sales.head(30), width="stretch", hide_index=True)
    if st.session_state.source_tables:
        with st.expander("Các bảng nguồn đã nạp"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Bảng": name, "Số dòng": len(frame), "Số cột": len(frame.columns)}
                        for name, frame in st.session_state.source_tables.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
