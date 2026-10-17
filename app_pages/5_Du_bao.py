# -*- coding: utf-8 -*-
"""Generate, inspect and export multi-step forecasts."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import configure_page, initialize_session_state
from model import forecast_future


configure_page("Dự báo doanh số")
initialize_session_state()

st.title("Dự báo doanh số")

model = st.session_state.model
scaler = st.session_state.scaler
scaled = st.session_state.get("full_scaled")
series = st.session_state.get("training_series")
if model is None or scaler is None or scaled is None or series is None:
    st.warning("Hãy huấn luyện mô hình ở trang **Huấn luyện mô hình** trước.")
    st.stop()

config = st.session_state.model_config
st.caption(
    f"Mô hình hiện tại: {config['type']} — look-back {config['look_back']} ngày; "
    f"chuỗi đã huấn luyện đến {series['date'].max():%d/%m/%Y}."
)
horizon = st.slider("Số ngày cần dự báo", 1, 90, 30)
if st.button("Tạo dự báo", type="primary"):
    try:
        values = forecast_future(
            model,
            scaled[-st.session_state.look_back :],
            scaler,
            horizon,
        )
        dates = pd.date_range(
            series["date"].max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        st.session_state.forecast_result = pd.DataFrame(
            {"date": dates, "forecast": values}
        )
        st.session_state.inventory_plan = None
    except ValueError as exc:
        st.error(f"Không thể tạo dự báo: {exc}")

forecast = st.session_state.forecast_result
if forecast is not None:
    history = series.tail(180)
    figure = go.Figure()
    figure.add_scatter(
        x=history["date"], y=history["sales"], mode="lines", name="Doanh số lịch sử"
    )
    figure.add_scatter(
        x=forecast["date"],
        y=forecast["forecast"],
        mode="lines+markers",
        name="Dự báo",
        line={"dash": "dash", "color": "#f97316"},
    )
    figure.update_layout(
        title="Doanh số lịch sử và dự báo",
        xaxis_title="Ngày",
        yaxis_title="Số lượng",
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")

    cols = st.columns(3)
    cols[0].metric("Tổng lượng dự báo", f"{forecast['forecast'].sum():,.1f}")
    cols[1].metric("Trung bình/ngày", f"{forecast['forecast'].mean():,.1f}")
    peak = forecast.loc[forecast["forecast"].idxmax()]
    cols[2].metric("Ngày dự báo cao nhất", f"{peak['date']:%d/%m/%Y}")
    st.dataframe(forecast, width="stretch", hide_index=True)
    st.download_button(
        "Tải dự báo CSV",
        forecast.to_csv(index=False).encode("utf-8-sig"),
        "du_bao_doanh_so.csv",
        "text/csv",
    )

st.divider()
st.subheader("Dự báo cho các dòng trong bảng test")
test_data = st.session_state.get("test_df")
training_filter = st.session_state.get("training_filter") or {}
selected_store = training_filter.get("store_nbr", "Tất cả cửa hàng")
selected_family = training_filter.get("family", "Tất cả nhóm hàng")
can_map_test_rows = (
    test_data is not None
    and selected_store != "Tất cả cửa hàng"
    and selected_family != "Tất cả nhóm hàng"
)

if test_data is None or test_data.empty:
    st.info("Chưa nạp bảng test.csv.")
elif not can_map_test_rows:
    st.info(
        "Để xuất dự báo có khóa `id`, hãy huấn luyện riêng một cửa hàng và một "
        "nhóm hàng. Mô hình tổng hợp nhiều chuỗi không thể gán dự báo chính xác "
        "cho từng dòng test."
    )
else:
    test_rows = test_data[
        (test_data["store_nbr"].astype(str) == str(selected_store))
        & (test_data["family"].astype(str) == str(selected_family))
    ].copy()
    test_rows = test_rows.dropna(subset=["date"]).sort_values("date")
    if test_rows.empty:
        st.info("Không có dòng test khớp cửa hàng/nhóm hàng đã dùng để huấn luyện.")
    else:
        st.caption(
            f"{len(test_rows):,} dòng test khớp cửa hàng {selected_store}, "
            f"nhóm hàng {selected_family}."
        )
        if st.button("Dự báo các dòng test"):
            first_date = pd.Timestamp(series["date"].max()) + pd.Timedelta(days=1)
            last_test_date = pd.Timestamp(test_rows["date"].max())
            days_to_predict = (last_test_date - pd.Timestamp(series["date"].max())).days
            if days_to_predict < 1:
                st.error("Ngày trong test.csv không nằm sau ngày cuối tập huấn luyện.")
            else:
                try:
                    values = forecast_future(
                        model,
                        scaled[-st.session_state.look_back :],
                        scaler,
                        days_to_predict,
                    )
                    date_predictions = pd.DataFrame(
                        {
                            "date": pd.date_range(first_date, periods=days_to_predict, freq="D"),
                            "sales": values,
                        }
                    )
                    test_result = test_rows.merge(
                        date_predictions, on="date", how="left", validate="many_to_one"
                    )
                    output = test_result.dropna(subset=["sales"])[["id", "sales"]]
                    template = st.session_state.source_tables.get("sample_submission")
                    if template is not None and "id" in template:
                        output = template[["id"]].merge(
                            output, on="id", how="inner", validate="one_to_one"
                        )
                    st.session_state.submission_result = output
                except ValueError as exc:
                    st.error(f"Không thể dự báo test: {exc}")

        submission = st.session_state.get("submission_result")
        if submission is not None:
            st.dataframe(submission.head(30), width="stretch", hide_index=True)
            st.download_button(
                "Tải dự báo test theo id",
                submission.to_csv(index=False).encode("utf-8-sig"),
                "test_predictions.csv",
                "text/csv",
            )
