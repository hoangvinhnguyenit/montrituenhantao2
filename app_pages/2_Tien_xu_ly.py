# -*- coding: utf-8 -*-
"""Missing-value, duplicate, outlier and feature-engineering workbench."""

import plotly.express as px
import streamlit as st

from app_common import configure_page, initialize_session_state, show_data_required
from utils import clean_sales_data, engineer_features


configure_page("Tiền xử lý & đặc trưng")
initialize_session_state()

st.title("Tiền xử lý & Feature Engineering")
st.caption(
    "Làm sạch dữ liệu có kiểm soát; tạo lag, trung bình trượt và đặc trưng lịch "
    "trước khi đưa vào mô hình."
)

raw = st.session_state.raw_df
if show_data_required(raw):
    st.stop()

missing = raw.isna().sum().rename("Số giá trị thiếu").to_frame()
missing["Tỷ lệ thiếu (%)"] = (missing["Số giá trị thiếu"] / len(raw) * 100).round(2)
left, right = st.columns([1, 1.4])
with left:
    st.subheader("Chất lượng dữ liệu nguồn")
    st.dataframe(missing, width="stretch")
    duplicate_key = ["id"] if "id" in raw else None
    duplicate_count = raw.duplicated(subset=duplicate_key).sum()
    st.metric("Dòng trùng khóa / bản ghi", f"{duplicate_count:,}")
with right:
    st.subheader("Phân phối doanh số & outlier")
    sales = raw["sales"].dropna()
    if not sales.empty:
        q1, q3 = sales.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((sales < q1 - 1.5 * iqr) | (sales > q3 + 1.5 * iqr)).sum())
        st.metric("Outlier theo ngưỡng 1.5×IQR", f"{outliers:,}")
        st.plotly_chart(
            px.box(raw, y="sales", points=False, title="Phân phối sales"),
            width="stretch",
        )

st.divider()
st.subheader("Tùy chọn làm sạch")
col1, col2, col3 = st.columns(3)
fill_missing = col1.checkbox("Thay sales thiếu bằng 0", value=True)
remove_duplicates = col2.checkbox("Loại dòng trùng ID / bản ghi giống hệt", value=True)
cap_outliers = col3.checkbox("Winsorize sales ở phân vị 1%–99%", value=False)
if st.button("Làm sạch và tạo đặc trưng", type="primary"):
    try:
        cleaned = clean_sales_data(
            raw,
            fill_missing_sales=fill_missing,
            remove_duplicates=remove_duplicates,
            cap_outliers=cap_outliers,
        )
        st.session_state.processed_df = engineer_features(cleaned)
        st.success(
            f"Hoàn tất: {len(cleaned):,} dòng sau làm sạch. "
            "Doanh số nguồn vẫn được giữ nguyên."
        )
    except ValueError as exc:
        st.error(f"Không thể xử lý dữ liệu: {exc}")

processed = st.session_state.processed_df
if processed is not None:
    st.subheader("Dữ liệu đã xử lý và đặc trưng tạo mới")
    st.caption(
        "lag_1/7/28 và rolling_mean_7/28 chỉ dùng các giá trị trước ngày hiện tại "
        "để tránh rò rỉ dữ liệu tương lai."
    )
    st.dataframe(processed.head(50), width="stretch", hide_index=True)
    csv = processed.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải dữ liệu đã xử lý (CSV)",
        csv,
        "du_lieu_da_xu_ly.csv",
        "text/csv",
    )
