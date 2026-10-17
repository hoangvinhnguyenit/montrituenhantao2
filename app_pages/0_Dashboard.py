# -*- coding: utf-8 -*-
"""Sales and inventory overview dashboard."""

import plotly.express as px
import streamlit as st

from app_common import (
    configure_page,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from utils import aggregate_daily


configure_page("Dashboard")
initialize_session_state()

st.title("Tổng quan doanh số, cửa hàng và tồn kho")
st.caption(
    "Theo dõi doanh số, cửa hàng và trạng thái tồn kho từ dữ liệu bán hàng."
)

sales = get_active_sales()
if show_data_required(sales):
    st.info("Mở trang **Dữ liệu** ở thanh bên để tải bộ dữ liệu hoặc tạo dữ liệu mẫu.")
else:
    forecast = st.session_state.get("forecast_result")
    metrics = st.session_state.get("test_metrics")
    inventory = st.session_state.get("inventory_plan")

    total_sales = float(sales["sales"].fillna(0).sum())
    stores_count = (
        int(sales["store_nbr"].nunique())
        if "store_nbr" in sales.columns
        else 0
    )
    families_count = int(sales["family"].nunique())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng lượng bán", f"{total_sales:,.0f}")
    col2.metric("Cửa hàng", f"{stores_count:,}")
    col3.metric("Nhóm hàng", f"{families_count:,}")
    col4.metric(
        "Dự báo sắp tới",
        f"{forecast['forecast'].sum():,.0f}" if forecast is not None else "Chưa có",
    )

    left, right = st.columns([1.6, 1])
    with left:
        daily = aggregate_daily(sales)
        fig = px.line(
            daily,
            x="date",
            y="sales",
            title="Doanh số theo ngày",
            labels={"date": "Ngày", "sales": "Số lượng bán"},
        )
        if forecast is not None and not forecast.empty:
            fig.add_scatter(
                x=forecast["date"],
                y=forecast["forecast"],
                mode="lines",
                name="Dự báo",
                line={"dash": "dash", "color": "#f97316"},
            )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Trạng thái mô hình")
        if metrics:
            st.metric("RMSE kiểm định", f"{metrics['RMSE']:,.2f}")
            st.metric("MAE kiểm định", f"{metrics['MAE']:,.2f}")
            st.metric("MAPE kiểm định", f"{metrics['MAPE (%)']:,.2f}%")
        else:
            st.info("Chưa có kết quả đánh giá. Huấn luyện mô hình để bắt đầu.")

        st.subheader("Tồn kho")
        if inventory:
            st.write(f"Điểm đặt hàng lại (ROP): **{inventory['reorder_point']:,.1f}**")
            st.write(
                "Khuyến nghị: **"
                + ("Cần đặt hàng" if inventory["need_reorder"] else "Chưa cần đặt hàng")
                + "**"
            )
        else:
            st.caption("Chưa có tham số tồn kho được tính.")

    st.subheader("Phân bổ doanh số theo nhóm hàng")
    family_sales = (
        sales.groupby("family", dropna=False)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .rename("sales")
        .reset_index()
    )
    st.plotly_chart(
        px.bar(
            family_sales,
            x="sales",
            y="family",
            orientation="h",
            labels={"sales": "Số lượng bán", "family": "Nhóm hàng"},
        ),
        width="stretch",
    )
