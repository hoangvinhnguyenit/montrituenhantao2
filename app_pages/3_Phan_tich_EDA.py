# -*- coding: utf-8 -*-
"""Exploratory analysis of store, family, promotion and calendar effects."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    configure_page,
    filter_controls,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from utils import aggregate_daily


configure_page("Phân tích EDA")
initialize_session_state()

st.title("Phân tích dữ liệu (EDA)")
st.caption(
    "Khám phá xu hướng, mùa vụ, cửa hàng, nhóm sản phẩm, khuyến mãi, "
    "ngày lễ và giá dầu."
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()
filtered = filter_controls(sales, key_prefix="eda")

minimum_date = filtered["date"].min().date()
maximum_date = filtered["date"].max().date()
date_range = st.date_input(
    "Khoảng thời gian",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[
        (filtered["date"] >= start_date) & (filtered["date"] < end_date)
    ]

daily = aggregate_daily(filtered)
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Tổng doanh số", f"{filtered['sales'].sum():,.0f}")
metric2.metric("Doanh số TB/ngày", f"{daily['sales'].mean():,.1f}")
metric3.metric(
    "Ngày cao điểm",
    f"{daily.loc[daily['sales'].idxmax(), 'date']:%d/%m/%Y}"
    if not daily.empty
    else "—",
)

tab_trend, tab_structure, tab_features = st.tabs(
    ["Xu hướng & mùa vụ", "Nhóm hàng & cửa hàng", "Khuyến mãi & yếu tố ngoài"]
)
with tab_trend:
    frequency = st.radio(
        "Tần suất biểu đồ",
        ["Ngày", "Tuần", "Tháng"],
        horizontal=True,
        key="eda_frequency",
    )
    period = {"Ngày": "D", "Tuần": "W", "Tháng": "MS"}[frequency]
    chart_data = aggregate_daily(filtered, freq=period)
    chart_data["Trung bình trượt"] = chart_data["sales"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=chart_data["date"], y=chart_data["sales"], name="Doanh số")
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["Trung bình trượt"],
            name="Trung bình trượt 7 kỳ",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(xaxis_title="Ngày", yaxis_title="Số lượng bán", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    weekday = filtered.assign(
        weekday=filtered["date"].dt.dayofweek.map(
            {0: "Thứ 2", 1: "Thứ 3", 2: "Thứ 4", 3: "Thứ 5", 4: "Thứ 6", 5: "Thứ 7", 6: "Chủ nhật"}
        )
    )
    weekday_sales = weekday.groupby("weekday", as_index=False)["sales"].mean()
    st.plotly_chart(
        px.bar(weekday_sales, x="weekday", y="sales", title="Doanh số trung bình theo thứ"),
        width="stretch",
    )

with tab_structure:
    if "family" in filtered:
        family_sales = (
            filtered.groupby("family", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                family_sales.head(20),
                x="sales",
                y="family",
                orientation="h",
                title="Top nhóm hàng theo doanh số",
            ),
            width="stretch",
        )
    if "store_nbr" in filtered:
        store_sales = (
            filtered.groupby("store_nbr", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )
        st.plotly_chart(
            px.bar(store_sales, x="store_nbr", y="sales", title="Doanh số theo cửa hàng"),
            width="stretch",
        )
    if {"date", "family"}.issubset(filtered.columns):
        heatmap = (
            filtered.assign(month=filtered["date"].dt.to_period("M").astype(str))
            .groupby(["month", "family"], as_index=False)["sales"]
            .sum()
        )
        st.plotly_chart(
            px.density_heatmap(
                heatmap,
                x="month",
                y="family",
                z="sales",
                title="Nhiệt đồ doanh số theo tháng và nhóm hàng",
                color_continuous_scale="Blues",
            ),
            width="stretch",
        )

with tab_features:
    if "onpromotion" in filtered:
        promo = filtered.assign(
            Trạng_thái=filtered["onpromotion"].fillna(0).gt(0).map(
                {True: "Có khuyến mãi", False: "Không khuyến mãi"}
            )
        )
        promo_summary = promo.groupby("Trạng_thái", as_index=False)["sales"].mean()
        st.plotly_chart(
            px.bar(
                promo_summary,
                x="Trạng_thái",
                y="sales",
                title="Doanh số trung bình theo trạng thái khuyến mãi",
            ),
            width="stretch",
        )
    if "has_holiday" in filtered:
        holiday_summary = (
            filtered.assign(
                Loại_ngày=filtered["has_holiday"].gt(0).map(
                    {True: "Ngày lễ/sự kiện", False: "Ngày thường"}
                )
            )
            .groupby("Loại_ngày", as_index=False)["sales"]
            .mean()
        )
        st.plotly_chart(
            px.bar(holiday_summary, x="Loại_ngày", y="sales", title="Ngày lễ và ngày thường"),
            width="stretch",
        )
    if "dcoilwtico" in filtered:
        external = filtered.dropna(subset=["dcoilwtico"])
        if not external.empty:
            st.plotly_chart(
                px.scatter(
                    external.sample(min(len(external), 5000), random_state=42),
                    x="dcoilwtico",
                    y="sales",
                    title="Tương quan giá dầu WTI và doanh số",
                ),
                width="stretch",
            )
    if "dcoilwtico" not in filtered and "has_holiday" not in filtered:
        st.info("Nạp thêm bảng oil.csv/holidays_events.csv để phân tích biến ngoại sinh.")

st.subheader("Thống kê mô tả và outlier")
q1, q3 = filtered["sales"].quantile([0.25, 0.75])
iqr = q3 - q1
outlier_rows = filtered[
    (filtered["sales"] < q1 - 1.5 * iqr)
    | (filtered["sales"] > q3 + 1.5 * iqr)
]
left, right = st.columns([1, 1])
left.dataframe(filtered["sales"].describe().to_frame().T, width="stretch")
right.metric("Số điểm ngoài ngưỡng 1.5×IQR", f"{len(outlier_rows):,}")
