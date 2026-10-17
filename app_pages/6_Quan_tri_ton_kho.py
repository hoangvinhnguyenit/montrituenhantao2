# -*- coding: utf-8 -*-
"""Demand-based replenishment and ABC product classification."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import configure_page, get_active_sales, initialize_session_state
from inventory import abc_classification, build_inventory_plan


configure_page("Quản trị tồn kho")
initialize_session_state()

st.title("Quản trị tồn kho")
st.caption(
    "ROP, safety stock và EOQ được ước tính từ dự báo. ERD hiện không có bảng "
    "tồn kho, vì vậy tồn kho hiện tại được nhập thủ công."
)

forecast = st.session_state.forecast_result
sales = get_active_sales()
if forecast is None or forecast.empty:
    st.warning("Hãy tạo dự báo ở trang **Dự báo doanh số** trước.")
    st.stop()

control1, control2, control3 = st.columns(3)
current_stock = control1.number_input(
    "Tồn kho hiện tại (đơn vị)", min_value=0.0, value=200.0, step=10.0
)
lead_time = control2.number_input(
    "Lead time (ngày)", min_value=1, max_value=365, value=7
)
service_level = control3.slider(
    "Mức độ phục vụ", min_value=0.80, max_value=0.99, value=0.95, step=0.01
)
cost1, cost2 = st.columns(2)
order_cost = cost1.number_input("Chi phí mỗi lần đặt hàng", min_value=0.0, value=0.0, step=1000.0)
holding_cost = cost2.number_input(
    "Chi phí lưu kho/đơn vị/năm", min_value=0.0, value=0.0, step=100.0
)

training_series = st.session_state.get("training_series")
historical_demand = None
if training_series is not None and not training_series.empty:
    historical_demand = (
        training_series.set_index("date")["sales"].asfreq("D", fill_value=0).tail(90).to_numpy()
    )
try:
    plan = build_inventory_plan(
        forecast["forecast"],
        current_stock=current_stock,
        lead_time=int(lead_time),
        service_level=service_level,
        order_cost=order_cost,
        holding_cost=holding_cost,
        historical_demand=historical_demand,
    )
    st.session_state.inventory_plan = plan
except ValueError as exc:
    st.error(f"Không thể tính kế hoạch tồn kho: {exc}")
    st.stop()

result_cols = st.columns(4)
result_cols[0].metric("Nhu cầu TB/ngày", f"{plan['avg_daily_demand']:,.1f}")
result_cols[1].metric("Tồn kho an toàn", f"{plan['safety_stock']:,.1f}")
result_cols[2].metric("Điểm đặt hàng lại (ROP)", f"{plan['reorder_point']:,.1f}")
result_cols[3].metric(
    "EOQ",
    f"{plan['eoq']:,.1f}" if plan["eoq"] is not None else "Nhập chi phí",
)
status_cols = st.columns(3)
status_cols[0].metric(
    "Trạng thái",
    "Cần đặt hàng" if plan["need_reorder"] else "Chưa cần đặt hàng",
)
status_cols[1].metric("Lượng đề xuất đặt", f"{plan['suggested_order_qty']:,.1f}")
days = plan["days_of_supply"]
status_cols[2].metric(
    "Số ngày cung ứng",
    f"{days:,.1f}" if days != float("inf") else "Không giới hạn",
)

projected = [float(current_stock)]
for demand in forecast["forecast"]:
    projected.append(max(projected[-1] - float(demand), 0))
chart = go.Figure()
chart.add_scatter(
    x=[0] + list(range(1, len(projected))),
    y=projected,
    mode="lines+markers",
    name="Tồn kho mô phỏng",
)
chart.add_hline(
    y=plan["reorder_point"],
    line_dash="dash",
    line_color="#dc2626",
    annotation_text="ROP",
)
chart.add_hline(
    y=plan["safety_stock"],
    line_dash="dot",
    line_color="#f97316",
    annotation_text="Tồn kho an toàn",
)
chart.update_layout(xaxis_title="Ngày", yaxis_title="Tồn kho ước tính")
st.plotly_chart(chart, width="stretch")

st.divider()
st.subheader("Phân loại ABC nhóm hàng")
if sales is not None and not sales.empty:
    abc = abc_classification(sales)
    st.dataframe(abc, width="stretch", hide_index=True)
    st.plotly_chart(
        px.bar(
            abc,
            x="family",
            y="sales",
            color="abc_class",
            color_discrete_map={"A": "#dc2626", "B": "#f97316", "C": "#16a34a"},
            title="Doanh số và phân loại ABC",
        ),
        width="stretch",
    )
    st.caption(
        "A: nhóm đóng góp xấp xỉ 80% doanh số; B: đến 95%; C: phần còn lại. "
        "Ngưỡng phân nhóm dựa trên tỷ trọng doanh số tích lũy."
    )
