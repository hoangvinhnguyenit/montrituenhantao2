# -*- coding: utf-8 -*-
"""Configure and train the LSTM/GRU sales forecasting model."""

import plotly.graph_objects as go
import streamlit as st

from app_common import (
    configure_page,
    filter_controls,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from model import train_recurrent_model
from utils import aggregate_daily


configure_page("Huấn luyện mô hình")
initialize_session_state()

st.title("Huấn luyện mô hình Deep Learning")
st.caption("Huấn luyện LSTM hoặc GRU cho chuỗi doanh số đã chọn.")

sales = get_active_sales()
if show_data_required(sales):
    st.stop()
filtered = filter_controls(sales, key_prefix="training")
daily = aggregate_daily(filtered)

if len(daily) < 20:
    st.error("Cần tối thiểu 20 ngày dữ liệu sau khi lọc để huấn luyện.")
    st.stop()

config_col, architecture_col = st.columns([1, 1.25])
with config_col:
    model_type = st.selectbox("Kiến trúc", ["LSTM", "GRU"])
    max_look_back = min(60, max(7, (len(daily) - 2) // 2))
    look_back = st.slider(
        "Cửa sổ nhìn lại (ngày)",
        min_value=7,
        max_value=max_look_back,
        value=min(28, max_look_back),
    )
    test_ratio = st.slider("Tỷ lệ kiểm định theo thời gian", 0.1, 0.4, 0.2, 0.05)
    units = st.select_slider("Units", options=[16, 32, 64, 128], value=64)
    layers = st.slider("Số lớp RNN", 1, 3, 2)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    epochs = st.slider("Epoch tối đa", 5, 100, 30, 5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    train_button = st.button("Bắt đầu huấn luyện", type="primary", width="stretch")

with architecture_col:
    st.subheader("Dữ liệu huấn luyện")
    st.metric("Số ngày", f"{len(daily):,}")
    st.write(
        f"**Từ:** {daily['date'].min():%d/%m/%Y}  \n"
        f"**Đến:** {daily['date'].max():%d/%m/%Y}  \n"
        f"**Tổng doanh số:** {daily['sales'].sum():,.0f}"
    )
    st.code(
        f"Input ({look_back}, 1)\n"
        + " -> ".join(
            [f"{model_type}({units})"] * layers
            + [f"Dropout({dropout})", "Dense(32, relu)", "Dense(1)"]
        )
        + "\nOptimizer: Adam | Loss: MSE",
        language="text",
    )
    st.line_chart(daily.set_index("date")["sales"])

if train_button:
    progress = st.progress(0, text="Chuẩn bị dữ liệu...")

    def update_progress(epoch, total, logs):
        percent = min(int(epoch / total * 100), 100)
        loss = logs.get("loss")
        progress.progress(
            percent,
            text=f"Epoch {epoch}/{total} — loss: {loss:.5f}" if loss is not None else f"Epoch {epoch}/{total}",
        )

    try:
        result = train_recurrent_model(
            daily,
            model_type=model_type,
            look_back=look_back,
            units=units,
            layers=layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            test_ratio=test_ratio,
            progress_callback=update_progress,
        )
        st.session_state.model = result["model"]
        st.session_state.scaler = result["scaler"]
        st.session_state.full_scaled = result["full_scaled"]
        st.session_state.look_back = result["look_back"]
        st.session_state.test_metrics = result["metrics"]
        st.session_state.test_pred_data = result["test_data"]
        st.session_state.training_history = result["history"]
        st.session_state.training_series = result["daily"]
        st.session_state.model_config = result["model_config"]
        st.session_state.training_filter = {
            "store_nbr": st.session_state.selected_store,
            "family": st.session_state.selected_family,
        }
        st.session_state.forecast_result = None
        st.session_state.submission_result = None
        st.session_state.inventory_plan = None
        progress.progress(100, text="Huấn luyện hoàn tất.")
        st.success("Mô hình đã được huấn luyện và đánh giá trên đoạn thời gian giữ lại.")
    except (ValueError, RuntimeError) as exc:
        st.error(f"Không thể huấn luyện: {exc}")

history = st.session_state.training_history
metrics = st.session_state.test_metrics
prediction = st.session_state.test_pred_data
if history is not None and metrics is not None and prediction is not None:
    st.divider()
    st.subheader("Kết quả kiểm định")
    metric_cols = st.columns(4)
    metric_cols[0].metric("MAE", f"{metrics['MAE']:,.2f}")
    metric_cols[1].metric("RMSE", f"{metrics['RMSE']:,.2f}")
    metric_cols[2].metric("MAPE", f"{metrics['MAPE (%)']:,.2f}%")
    metric_cols[3].metric("WAPE", f"{metrics['WAPE (%)']:,.2f}%")
    chart1, chart2 = st.columns(2)
    with chart1:
        loss_chart = go.Figure()
        loss_chart.add_scatter(y=history["loss"], mode="lines", name="Train loss")
        if "val_loss" in history:
            loss_chart.add_scatter(y=history["val_loss"], mode="lines", name="Validation loss")
        loss_chart.update_layout(title="Loss theo epoch", xaxis_title="Epoch", yaxis_title="MSE")
        st.plotly_chart(loss_chart, width="stretch")
    with chart2:
        prediction_chart = go.Figure()
        prediction_chart.add_scatter(
            x=prediction["dates"], y=prediction["actual"], mode="lines", name="Thực tế"
        )
        prediction_chart.add_scatter(
            x=prediction["dates"],
            y=prediction["predicted"],
            mode="lines",
            name="Dự báo",
        )
        prediction_chart.update_layout(title="Thực tế và dự báo trên tập kiểm định")
        st.plotly_chart(prediction_chart, width="stretch")
    st.caption(f"Cấu hình: {st.session_state.model_config}")
