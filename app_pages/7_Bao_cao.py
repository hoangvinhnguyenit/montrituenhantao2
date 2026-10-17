# -*- coding: utf-8 -*-
"""Consolidated analysis, model metrics and downloadable outputs."""

import pandas as pd
import streamlit as st

from app_common import configure_page, get_active_sales, initialize_session_state
from inventory import abc_classification


configure_page("Báo cáo tổng hợp")
initialize_session_state()

st.title("Báo cáo tổng hợp")

sales = get_active_sales()
if sales is None or sales.empty:
    st.warning("Chưa có dữ liệu. Nạp dữ liệu trước ở trang **Dữ liệu**.")
    st.stop()

st.subheader("Tổng quan dữ liệu")
columns = st.columns(4)
columns[0].metric("Số bản ghi", f"{len(sales):,}")
columns[1].metric("Nhóm hàng", f"{sales['family'].nunique():,}")
columns[2].metric(
    "Cửa hàng",
    f"{sales['store_nbr'].nunique():,}" if "store_nbr" in sales else "—",
)
columns[3].metric(
    "Khoảng ngày",
    f"{(sales['date'].max() - sales['date'].min()).days:,} ngày",
)

metrics = st.session_state.test_metrics
st.subheader("Đánh giá mô hình")
if metrics is None:
    st.info("Chưa có kết quả. Huấn luyện mô hình ở trang **Huấn luyện mô hình**.")
else:
    metric_cols = st.columns(4)
    for column, name in zip(metric_cols, ("MAE", "RMSE", "MAPE (%)", "WAPE (%)")):
        value = metrics[name]
        column.metric(name, f"{value:,.2f}" if pd.notna(value) else "Không xác định")
    config = st.session_state.model_config
    st.caption(f"Cấu hình: {config}")

st.subheader("Dự báo")
forecast = st.session_state.forecast_result
if forecast is None:
    st.info("Chưa có dự báo. Tạo dự báo ở trang **Dự báo doanh số**.")
else:
    st.dataframe(forecast, width="stretch", hide_index=True)
    st.download_button(
        "Tải forecast.csv",
        forecast.to_csv(index=False).encode("utf-8-sig"),
        "forecast.csv",
        "text/csv",
    )

st.subheader("Phân loại ABC")
abc = abc_classification(sales)
st.dataframe(abc, width="stretch", hide_index=True)
st.download_button(
    "Tải abc_classification.csv",
    abc.to_csv(index=False).encode("utf-8-sig"),
    "abc_classification.csv",
    "text/csv",
)

st.subheader("Xuất dữ liệu")
st.download_button(
    "Tải dữ liệu hiện hành",
    sales.to_csv(index=False).encode("utf-8-sig"),
    "sales_data.csv",
    "text/csv",
)
