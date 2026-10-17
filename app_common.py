# -*- coding: utf-8 -*-
"""Shared Streamlit setup and helpers used by independent pages."""

import streamlit as st


STATE_DEFAULTS = {
    "raw_df": None,
    "processed_df": None,
    "test_df": None,
    "source_tables": {},
    "source_label": None,
    "model": None,
    "scaler": None,
    "look_back": 14,
    "test_metrics": None,
    "test_pred_data": None,
    "training_history": None,
    "forecast_result": None,
    "submission_result": None,
    "model_config": None,
    "training_filter": None,
    "training_series": None,
    "full_scaled": None,
    "inventory_plan": None,
    "selected_store": "Tất cả cửa hàng",
    "selected_family": "Tất cả nhóm hàng",
}


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Dự báo doanh số",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_session_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_active_sales():
    initialize_session_state()
    processed = st.session_state.get("processed_df")
    return processed if processed is not None else st.session_state.get("raw_df")


def show_data_required(sales) -> bool:
    if sales is None or sales.empty:
        st.warning("Chưa có dữ liệu bán hàng. Hãy nạp dữ liệu ở trang **Dữ liệu**.")
        return True
    return False


def filter_controls(data, *, key_prefix: str):
    """Render shared store/family filters and return the filtered fact table."""
    initialize_session_state()
    frame = data.copy()
    col_store, col_family = st.columns(2)
    selected_store = st.session_state.selected_store
    selected_family = st.session_state.selected_family

    if "store_nbr" in frame:
        store_options = ["Tất cả cửa hàng"] + sorted(
            frame["store_nbr"].dropna().astype(str).unique().tolist()
        )
        if selected_store not in store_options:
            selected_store = store_options[0]
        selected_store = col_store.selectbox(
            "Cửa hàng",
            store_options,
            index=store_options.index(selected_store),
            key=f"{key_prefix}_store",
        )
        st.session_state.selected_store = selected_store
        if selected_store != "Tất cả cửa hàng":
            frame = frame[frame["store_nbr"].astype(str) == selected_store]

    if "family" in frame:
        family_options = ["Tất cả nhóm hàng"] + sorted(
            frame["family"].dropna().astype(str).unique().tolist()
        )
        if selected_family not in family_options:
            selected_family = family_options[0]
        selected_family = col_family.selectbox(
            "Nhóm hàng",
            family_options,
            index=family_options.index(selected_family),
            key=f"{key_prefix}_family",
        )
        st.session_state.selected_family = selected_family
        if selected_family != "Tất cả nhóm hàng":
            frame = frame[frame["family"].astype(str) == selected_family]
    return frame
