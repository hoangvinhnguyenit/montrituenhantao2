# -*- coding: utf-8 -*-
"""Inventory planning calculations derived from demand forecasts."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def build_inventory_plan(
    forecast,
    current_stock: float,
    lead_time: int,
    service_level: float,
    order_cost: float = 0,
    holding_cost: float = 0,
    historical_demand=None,
) -> dict:
    """Calculate safety stock, ROP, EOQ and replenishment recommendation."""
    expected = np.asarray(forecast, dtype=float).reshape(-1)
    expected = expected[np.isfinite(expected)]
    if expected.size == 0:
        raise ValueError("Cần có ít nhất một giá trị dự báo hợp lệ.")
    if current_stock < 0 or lead_time < 1:
        raise ValueError("Tồn kho không được âm và lead time phải từ 1 ngày.")
    if not 0 < service_level < 1:
        raise ValueError("Service level phải nằm trong khoảng (0, 1).")

    daily_mean = float(np.mean(expected))
    history = (
        np.asarray(historical_demand, dtype=float).reshape(-1)
        if historical_demand is not None
        else expected
    )
    history = history[np.isfinite(history)]
    daily_std = float(np.std(history, ddof=1)) if history.size > 1 else 0.0
    z_score = NormalDist().inv_cdf(service_level)
    safety_stock = z_score * daily_std * np.sqrt(lead_time)
    lead_time_demand = (
        float(expected[:lead_time].sum())
        if expected.size >= lead_time
        else daily_mean * lead_time
    )
    reorder_point = lead_time_demand + safety_stock
    annual_demand = daily_mean * 365
    eoq = (
        float(np.sqrt(2 * annual_demand * order_cost / holding_cost))
        if order_cost > 0 and holding_cost > 0 and annual_demand > 0
        else None
    )
    need_reorder = current_stock <= reorder_point
    target_stock = reorder_point + (eoq or lead_time_demand)
    projected = max(float(current_stock) - float(expected.sum()), 0.0)
    return {
        "avg_daily_demand": daily_mean,
        "daily_demand_std": daily_std,
        "safety_stock": float(max(safety_stock, 0)),
        "lead_time_demand": lead_time_demand,
        "reorder_point": float(max(reorder_point, 0)),
        "eoq": eoq,
        "need_reorder": need_reorder,
        "suggested_order_qty": float(max(target_stock - current_stock, 0))
        if need_reorder
        else 0.0,
        "days_of_supply": float(current_stock / daily_mean)
        if daily_mean > 0
        else float("inf"),
        "projected_ending_stock": projected,
    }


def abc_classification(data: pd.DataFrame) -> pd.DataFrame:
    """Classify product families into A/B/C from cumulative sales contribution."""
    if "family" not in data or "sales" not in data:
        raise ValueError("Dữ liệu cần có cột family và sales để phân loại ABC.")
    totals = (
        data.groupby("family", dropna=False)["sales"]
        .sum()
        .sort_values(ascending=False)
        .rename("sales")
        .reset_index()
    )
    grand_total = float(totals["sales"].sum())
    totals["cum_pct"] = (
        totals["sales"].cumsum() / grand_total * 100 if grand_total > 0 else 0
    )
    cumulative_before = totals["cum_pct"] - (
        totals["sales"] / grand_total * 100 if grand_total > 0 else 0
    )
    totals["abc_class"] = np.select(
        [cumulative_before < 80, cumulative_before < 95],
        ["A", "B"],
        default="C",
    )
    return totals
