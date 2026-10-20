# -*- coding: utf-8 -*-
"""Quản trị tồn kho: tình trạng hiện tại, mô phỏng tồn kho tương lai và rủi ro thiếu hàng.

Nhu cầu tương lai lấy từ kết quả dự báo của mô hình LSTM/GRU (trang Dự báo doanh số).
Độ bất định của nhu cầu được mô phỏng Monte Carlo quanh đường dự báo, dùng sai số
kiểm định (RMSE) của chính mô hình.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_common import configure_page, get_active_sales, initialize_session_state
from inventory import abc_classification, build_inventory_plan
from theme import CHART_COLORS, kpi_cards, page_intro, status_tag

configure_page("Quản trị tồn kho")
initialize_session_state()

ALL_STORE = "Tất cả cửa hàng"
ALL_FAMILY = "Tất cả nhóm hàng"
EPS = 1e-9


# ------------------------------------------------------------------ tiện ích
def _num(value: float) -> str:
    value = float(value)
    for unit, size in (("tỷ", 1e9), ("tr", 1e6)):
        if abs(value) >= size:
            return f"{value / size:,.1f} {unit}"
    return f"{value:,.0f}"


def _day(date) -> str:
    return f"{pd.Timestamp(date):%d/%m/%Y}"


def _first_true(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


@st.cache_data(show_spinner=False)
def compute_abc(_sales, signature, store):
    frame = _sales
    if store != ALL_STORE and "store_nbr" in frame:
        frame = frame[frame["store_nbr"].astype(str) == store]
    try:
        return abc_classification(frame)
    except ValueError:
        return pd.DataFrame()


def simulate(demand, stock0, lead, rop, target, incoming, replenish):
    """Mô phỏng tồn kho theo ngày (chính sách (s, S), bán thiếu thì mất đơn).

    demand: mảng (số kịch bản, số ngày). Đầu mỗi ngày: nhận hàng về, rồi nếu
    (tồn + hàng đang về) <= ROP thì đặt bù tới mức mục tiêu, hàng về sau `lead` ngày.
    """
    sims, horizon = demand.shape
    arrivals = np.zeros((sims, horizon + lead + 1))
    on_order = np.zeros(sims)
    in_qty, in_eta = incoming
    if in_qty > 0:
        arrivals[:, min(max(int(in_eta) - 1, 0), arrivals.shape[1] - 1)] += in_qty
        on_order += in_qty
    stock = np.full(sims, float(stock0))
    out = {k: np.zeros((sims, horizon)) for k in ("stock", "short", "orders", "received")}
    for d in range(horizon):
        received = arrivals[:, d]
        stock += received
        on_order -= received
        out["received"][:, d] = received
        if replenish:
            position = stock + on_order
            qty = np.where(position <= rop, np.maximum(target - position, 0.0), 0.0)
            on_order += qty
            out["orders"][:, d] = qty
            arrivals[:, d + lead] += qty
        sold = np.minimum(stock, demand[:, d])
        out["short"][:, d] = demand[:, d] - sold
        stock -= sold
        out["stock"][:, d] = stock
    return out


def scenario_stats(result, demand):
    short_total = result["short"].sum(axis=1)
    return {
        "p_any": float((short_total > EPS).mean()),
        "exp_short": float(short_total.mean()),
        "fill": float(1 - result["short"].sum() / max(demand.sum(), EPS)),
        "end_stock": float(result["stock"][:, -1].mean()),
        "n_orders": float((result["orders"] > EPS).sum(axis=1).mean()),
    }


def median_first_stockout(result, dates):
    flags = result["short"] > EPS
    any_short = flags.any(axis=1)
    if not any_short.any():
        return "—"
    first = np.argmax(flags[any_short], axis=1)
    return _day(dates[int(np.median(first))])


# ------------------------------------------------------------------ dữ liệu đầu vào
page_intro(
    "Quản trị tồn kho",
    "Tình trạng tồn kho hiện tại, dự đoán tồn kho và rủi ro thiếu hàng trong các ngày tới "
    "dựa trên dự báo của mô hình.",
)

forecast = st.session_state.forecast_result
sales = get_active_sales()
series = st.session_state.get("training_series")
if forecast is None or forecast.empty:
    st.warning("Hãy tạo dự báo ở trang **Dự báo doanh số** trước.")
    st.stop()

forecast = forecast.sort_values("date").reset_index(drop=True)
dates = pd.DatetimeIndex(forecast["date"])
demand_fc = np.maximum(forecast["forecast"].to_numpy(dtype=float), 0.0)
horizon = len(demand_fc)
last_date = (
    pd.Timestamp(series["date"].max()) if series is not None and not series.empty
    else dates[0] - pd.Timedelta(days=1)
)
scope = st.session_state.get("training_filter") or {}
store_scope = scope.get("store_nbr", ALL_STORE)
family_scope = scope.get("family", ALL_FAMILY)
scope_key = f"{store_scope}_{family_scope}"

st.caption(
    f"Phạm vi dự báo: **{'cửa hàng ' + store_scope if store_scope != ALL_STORE else 'tất cả cửa hàng'}** · "
    f"**{family_scope if family_scope != ALL_FAMILY else 'tất cả nhóm hàng'}** · "
    f"mốc hiện tại: {_day(last_date)} · dự báo {horizon} ngày "
    f"({_day(dates[0])} – {_day(dates[-1])}). ERD không có bảng tồn kho nên tồn kho hiện tại nhập thủ công."
)
if store_scope == ALL_STORE or family_scope == ALL_FAMILY:
    st.info(
        "Mô hình đang dự báo gộp nhiều cửa hàng/nhóm hàng, nên tồn kho nhập vào cần là "
        "tổng tồn của đúng phạm vi đó. Để ra quyết định đặt hàng chính xác, hãy huấn luyện "
        "riêng cho một cửa hàng và một nhóm hàng."
    )

sales_signature = (
    (len(sales), str(sales["date"].min()), str(sales["date"].max()),
     round(float(sales["sales"].fillna(0).sum()), 2))
    if sales is not None and not sales.empty else None
)
abc = compute_abc(sales, sales_signature, store_scope) if sales_signature else pd.DataFrame()
abc_class = None
if not abc.empty and family_scope != ALL_FAMILY:
    match = abc.loc[abc["family"].astype(str) == family_scope, "abc_class"]
    abc_class = match.iloc[0] if len(match) else None
default_service = {"A": 0.98, "B": 0.95, "C": 0.90}.get(abc_class, 0.95)

# ------------------------------------------------------------------ thông số
with st.container(border=True):
    st.subheader("Thông số tồn kho")
    c1, c2, c3, c4 = st.columns(4)
    default_stock = max(10.0, float(round(demand_fc.mean() * 10, -1)))
    current_stock = c1.number_input(
        "Tồn kho hiện tại (đơn vị)", min_value=0.0, value=default_stock, step=10.0,
        key=f"inv_stock_{scope_key}",
    )
    in_transit = c2.number_input(
        "Hàng đang về (đơn vị)", min_value=0.0, value=0.0, step=10.0, key=f"inv_transit_{scope_key}"
    )
    in_eta = c3.number_input(
        "Hàng về sau (ngày)", min_value=1, max_value=365, value=3,
        disabled=in_transit <= 0, key=f"inv_eta_{scope_key}",
    )
    lead_time = c4.number_input("Lead time đặt hàng (ngày)", min_value=1, max_value=365, value=7)
    s1, s2, s3 = st.columns(3)
    service_level = s1.slider(
        "Mức độ phục vụ", 0.80, 0.99, default_service, 0.01, key=f"inv_service_{abc_class}",
        help="Mặc định gợi ý theo hạng ABC: A 98%, B 95%, C 90%.",
    )
    order_cost = s2.number_input("Chi phí mỗi lần đặt hàng", min_value=0.0, value=0.0, step=1000.0)
    holding_cost = s3.number_input(
        "Chi phí lưu kho/đơn vị/năm", min_value=0.0, value=0.0, step=100.0
    )
    if abc_class:
        st.caption(f"Nhóm hàng **{family_scope}** thuộc hạng **{abc_class}** theo phân loại ABC.")

history = None
if series is not None and not series.empty:
    history = series.set_index("date")["sales"].asfreq("D", fill_value=0).tail(90).to_numpy()
try:
    plan = build_inventory_plan(
        demand_fc,
        current_stock=current_stock,
        lead_time=int(lead_time),
        service_level=service_level,
        order_cost=order_cost,
        holding_cost=holding_cost,
        historical_demand=history,
    )
except ValueError as exc:
    st.error(f"Không thể tính kế hoạch tồn kho: {exc}")
    st.stop()

rop, ss = plan["reorder_point"], plan["safety_stock"]
order_qty = plan["eoq"] or plan["lead_time_demand"]
target = rop + order_qty
plan["target_stock"] = target
plan["in_transit"] = float(in_transit)
st.session_state.inventory_plan = plan

metrics = st.session_state.get("test_metrics")
sigma_options = {}
if metrics and np.isfinite(metrics.get("RMSE", np.nan)):
    sigma_options["Sai số mô hình (RMSE kiểm định)"] = float(metrics["RMSE"])
sigma_options["Biến động nhu cầu lịch sử"] = float(plan["daily_demand_std"])
with st.expander("Tùy chọn mô phỏng nâng cao"):
    sigma_label = st.radio("Độ bất định của nhu cầu mỗi ngày", list(sigma_options), horizontal=True)
    n_sims = st.slider("Số kịch bản Monte Carlo", 200, 3000, 1000, 100)
    st.caption(
        "Mỗi kịch bản = nhu cầu dự báo + nhiễu chuẩn có độ lệch bằng giá trị đã chọn "
        "(không âm). Chính sách bổ sung: đầu ngày nếu tồn + hàng đang về ≤ ROP thì đặt bù "
        "tới mức mục tiêu; thiếu hàng được tính là mất đơn."
    )
sigma = sigma_options[sigma_label]

# ------------------------------------------------------------------ mô phỏng
rng = np.random.default_rng(42)
det_demand = demand_fc[None, :]
mc_demand = np.maximum(demand_fc[None, :] + sigma * rng.standard_normal((n_sims, horizon)), 0.0)
incoming = (float(in_transit), int(in_eta))
lead = int(lead_time)
det_plan = simulate(det_demand, current_stock, lead, rop, target, incoming, True)
det_none = simulate(det_demand, current_stock, lead, rop, target, incoming, False)
mc_plan = simulate(mc_demand, current_stock, lead, rop, target, incoming, True)
mc_none = simulate(mc_demand, current_stock, lead, rop, target, incoming, False)
stats_plan = scenario_stats(mc_plan, mc_demand)
stats_none = scenario_stats(mc_none, mc_demand)

stockout_idx = _first_true(det_none["short"][0] > EPS)
position = current_stock + in_transit
days_cover = plan["days_of_supply"]
recent = float(series["sales"].tail(28).mean()) if series is not None and len(series) else None
trend = (plan["avg_daily_demand"] / recent - 1) * 100 if recent else None

if current_stock <= 0:
    tag, level, headline = "Đã hết hàng", "bad", "Kho đang hết hàng."
elif position <= ss:
    tag, level, headline = "Dưới tồn kho an toàn", "bad", "Tồn kho đã xuống dưới mức an toàn."
elif position <= rop:
    tag, level, headline = "Cần đặt hàng", "warn", "Tồn kho đã chạm điểm đặt hàng lại (ROP)."
elif position > target * 1.5 and target > 0:
    tag, level, headline = "Tồn kho cao", "warn", "Tồn kho cao hơn nhiều so với mức mục tiêu."
else:
    tag, level, headline = "Đủ hàng", "ok", "Tồn kho đang ở mức an toàn."

# ------------------------------------------------------------------ HIỆN TẠI
st.header("1. Tình trạng hiện tại")
gauge_col, info_col = st.columns([1, 1.5])
with gauge_col:
    axis_max = max(target * 1.5, current_stock * 1.15, rop * 2, 1.0)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(current_stock),
        number={"suffix": " đv", "valueformat": ",.0f"},
        title={"text": "Tồn kho hiện tại"},
        gauge={
            "axis": {"range": [0, axis_max]},
            "bar": {"color": "#0f172a", "thickness": 0.28},
            "steps": [
                {"range": [0, ss], "color": "rgba(220,38,38,.35)"},
                {"range": [ss, rop], "color": "rgba(245,158,11,.35)"},
                {"range": [rop, target], "color": "rgba(16,185,129,.35)"},
                {"range": [target, axis_max], "color": "rgba(56,189,248,.30)"},
            ],
            "threshold": {"line": {"color": "#dc2626", "width": 4}, "thickness": 0.85, "value": rop},
        },
    ))
    gauge.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(gauge, width="stretch")
    st.caption("Đỏ: dưới tồn an toàn · Cam: dưới ROP · Xanh lá: vùng khỏe · Xanh dương: vượt mức mục tiêu. "
               "Vạch đỏ = ROP.")
with info_col:
    st.markdown(f"**Trạng thái:** {status_tag(tag, level)} &nbsp; {headline}", unsafe_allow_html=True)
    r1 = st.columns(3)
    r1[0].metric("Nhu cầu TB/ngày (dự báo)", f"{plan['avg_daily_demand']:,.1f}",
                 f"{trend:+.1f}% so với 28 ngày gần nhất" if trend is not None else None,
                 delta_color="off")
    r1[1].metric("Số ngày đủ bán", "∞" if not np.isfinite(days_cover) else f"{days_cover:,.1f}")
    r1[2].metric("Vị thế tồn (kho + đang về)", f"{position:,.0f}")
    r2 = st.columns(3)
    r2[0].metric("Tồn kho an toàn", f"{ss:,.1f}")
    r2[1].metric("Điểm đặt hàng lại (ROP)", f"{rop:,.1f}")
    r2[2].metric("Mức tồn mục tiêu", f"{target:,.1f}")
    r3 = st.columns(3)
    r3[0].metric("EOQ", f"{plan['eoq']:,.1f}" if plan["eoq"] else "Nhập chi phí")
    r3[1].metric("Nhu cầu trong lead time", f"{plan['lead_time_demand']:,.1f}")
    r3[2].metric("Đề xuất đặt ngay", f"{plan['suggested_order_qty']:,.0f}")
    if horizon < lead:
        st.warning(
            f"Số ngày dự báo ({horizon}) ngắn hơn lead time ({lead}); nhu cầu trong lead time "
            "được ước tính bằng mức trung bình. Hãy dự báo dài hơn để kết quả đáng tin cậy hơn."
        )

# ------------------------------------------------------------------ TƯƠNG LAI
st.header("2. Dự đoán tồn kho & thiếu hàng")

if stockout_idx is None:
    stockout_value, stockout_note = "Không hết", f"Đủ hàng trong {horizon} ngày dự báo (chưa bổ sung)"
    deadline_value, deadline_note = "Chưa cần", "Chưa có nguy cơ đứt hàng trong kỳ"
else:
    stockout_date = dates[stockout_idx]
    stockout_value = _day(stockout_date)
    stockout_note = f"Sau {stockout_idx + 1} ngày nếu không bổ sung"
    order_idx = stockout_idx - lead
    if order_idx < 0:
        deadline_value = "Đặt ngay"
        deadline_note = f"Đã trễ {-order_idx} ngày so với lead time {lead} ngày"
    else:
        deadline_value = _day(dates[order_idx])
        deadline_note = f"Để hàng về trước {_day(stockout_date)}"

kpi_cards([
    ("Dự kiến hết hàng (chưa bổ sung)", stockout_value, stockout_note),
    ("Hạn chót đặt hàng", deadline_value, deadline_note),
    ("Xác suất thiếu hàng trong kỳ", f"{stats_plan['p_any']:.0%}",
     f"Nếu không bổ sung: {stats_none['p_any']:.0%}"),
    ("Mức đáp ứng nhu cầu", f"{stats_plan['fill']:.1%}",
     f"Thiếu hụt TB {stats_plan['exp_short']:,.0f} đv (chưa bổ sung: {stats_none['fill']:.1%})"),
])

if stockout_idx is not None and stockout_idx < lead:
    st.error(
        f"Cảnh báo đứt hàng: nếu không có hàng về, kho sẽ hết vào {stockout_value}, "
        f"sớm hơn lead time {lead} ngày. Cần đặt hàng và xử lý khẩn cấp."
    )
elif plan["need_reorder"]:
    st.warning(
        f"Nên đặt hàng ngay khoảng {plan['suggested_order_qty']:,.0f} đơn vị "
        f"để đưa tồn về mức mục tiêu {target:,.0f}."
    )
else:
    st.success("Chưa cần đặt hàng ở thời điểm hiện tại; theo dõi lịch đặt hàng dự kiến bên dưới.")

tab_flow, tab_risk, tab_detail = st.tabs(
    ["Diễn biến tồn kho", "Rủi ro thiếu hàng", "Lịch đặt hàng & chi tiết"]
)

with tab_flow:
    x_all = [last_date] + list(dates)

    def with_start(values):
        return np.r_[float(current_stock), values]

    p10, p90 = np.percentile(mc_plan["stock"], [10, 90], axis=0)
    n10, n90 = np.percentile(mc_none["stock"], [10, 90], axis=0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.06)
    fig.add_hrect(y0=0, y1=ss, fillcolor="rgba(220,38,38,.07)", line_width=0, row=1, col=1)
    for upper, lower, color, label in (
        (n90, n10, "rgba(220,38,38,.12)", "Vùng P10–P90 (không bổ sung)"),
        (p90, p10, "rgba(16,185,129,.20)", "Vùng P10–P90 (có bổ sung)"),
    ):
        fig.add_scatter(x=x_all, y=with_start(upper), mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False, row=1, col=1)
        fig.add_scatter(x=x_all, y=with_start(lower), mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=color, name=label, hoverinfo="skip",
                        row=1, col=1)
    fig.add_scatter(x=x_all, y=with_start(det_none["stock"][0]), mode="lines",
                    name="Không bổ sung (dự báo TB)", line=dict(color="#dc2626", dash="dash", width=2.5),
                    row=1, col=1)
    fig.add_scatter(x=x_all, y=with_start(det_plan["stock"][0]), mode="lines",
                    name="Bổ sung theo ROP (dự báo TB)", line=dict(color="#10b981", width=3),
                    row=1, col=1)
    order_idx_list = np.flatnonzero(det_plan["orders"][0] > EPS)
    if order_idx_list.size:
        fig.add_scatter(
            x=dates[order_idx_list], y=det_plan["stock"][0][order_idx_list], mode="markers",
            name="Đặt hàng", marker=dict(symbol="diamond", size=12, color="#f59e0b",
                                         line=dict(color="white", width=1.5)),
            customdata=det_plan["orders"][0][order_idx_list],
            hovertemplate="Đặt hàng %{customdata:,.0f} đv<extra></extra>", row=1, col=1,
        )
    arrive_idx_list = np.flatnonzero(det_plan["received"][0] > EPS)
    if arrive_idx_list.size:
        fig.add_scatter(
            x=dates[arrive_idx_list], y=det_plan["stock"][0][arrive_idx_list], mode="markers",
            name="Hàng về", marker=dict(symbol="triangle-up", size=12, color="#38bdf8",
                                        line=dict(color="white", width=1.5)),
            customdata=det_plan["received"][0][arrive_idx_list],
            hovertemplate="Hàng về %{customdata:,.0f} đv<extra></extra>", row=1, col=1,
        )
    fig.add_scatter(x=[last_date], y=[float(current_stock)], mode="markers+text", text=["Hiện tại"],
                    textposition="top right", name="Hiện tại",
                    marker=dict(size=11, color="#0f172a"), showlegend=False, row=1, col=1)
    fig.add_hline(y=rop, line_dash="dash", line_color="#dc2626", annotation_text="ROP",
                  annotation_position="top left", row=1, col=1)
    fig.add_hline(y=ss, line_dash="dot", line_color="#f97316", annotation_text="Tồn an toàn",
                  annotation_position="bottom left", row=1, col=1)
    fig.add_bar(x=dates, y=demand_fc, name="Nhu cầu dự báo/ngày", marker_color=CHART_COLORS[1],
                row=2, col=1)
    fig.update_yaxes(title_text="Tồn kho (đơn vị)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Nhu cầu/ngày", row=2, col=1)
    fig.update_layout(height=640, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Tồn cuối kỳ dự kiến: **{det_plan['stock'][0][-1]:,.0f}** đv (có bổ sung) · "
        f"**{det_none['stock'][0][-1]:,.0f}** đv (không bổ sung). "
        f"Dải màu thể hiện 80% kịch bản Monte Carlo ({n_sims:,} kịch bản, σ = {sigma:,.1f}/ngày)."
    )

with tab_risk:
    p_day_plan = (mc_plan["short"] > EPS).mean(axis=0) * 100
    p_day_none = (mc_none["short"] > EPS).mean(axis=0) * 100
    risk = go.Figure()
    risk.add_scatter(x=dates, y=p_day_none, mode="lines", name="Không bổ sung",
                     line=dict(color="#dc2626", width=2.5), fill="tozeroy",
                     fillcolor="rgba(220,38,38,.10)")
    risk.add_scatter(x=dates, y=p_day_plan, mode="lines", name="Bổ sung theo ROP",
                     line=dict(color="#10b981", width=3))
    risk.update_layout(title="Xác suất thiếu hàng theo ngày", yaxis_title="% kịch bản bị thiếu",
                       yaxis=dict(range=[0, 100]), hovermode="x unified", height=380)
    st.plotly_chart(risk, width="stretch")

    comparison = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Xác suất thiếu hàng trong kỳ",
                "Ngày thiếu hàng đầu tiên (trung vị)",
                "Thiếu hụt trung bình (đv)",
                "Mức đáp ứng nhu cầu",
                "Tồn cuối kỳ trung bình (đv)",
                "Số lần đặt hàng trung bình",
            ],
            "Không bổ sung": [
                f"{stats_none['p_any']:.1%}", median_first_stockout(mc_none, dates),
                f"{stats_none['exp_short']:,.0f}", f"{stats_none['fill']:.1%}",
                f"{stats_none['end_stock']:,.0f}", "0",
            ],
            "Bổ sung theo ROP": [
                f"{stats_plan['p_any']:.1%}", median_first_stockout(mc_plan, dates),
                f"{stats_plan['exp_short']:,.0f}", f"{stats_plan['fill']:.1%}",
                f"{stats_plan['end_stock']:,.0f}", f"{stats_plan['n_orders']:.1f}",
            ],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        f"Mức phục vụ mục tiêu {service_level:.0%}: xác suất thiếu hàng trong kỳ khi bổ sung "
        f"{'đạt' if stats_plan['fill'] >= service_level else 'chưa đạt'} mục tiêu "
        f"(mức đáp ứng mô phỏng {stats_plan['fill']:.1%}). Nếu chưa đạt, hãy tăng mức phục vụ "
        "hoặc rút ngắn lead time."
    )

with tab_detail:
    orders_idx = np.flatnonzero(det_plan["orders"][0] > EPS)
    st.subheader("Lịch đặt hàng dự kiến")
    if orders_idx.size:
        schedule = pd.DataFrame(
            {
                "Ngày đặt": [_day(dates[i]) for i in orders_idx],
                "Số lượng đặt": det_plan["orders"][0][orders_idx].round(0),
                "Ngày hàng về": [_day(dates[i] + pd.Timedelta(days=lead)) for i in orders_idx],
                "Tồn dự kiến khi đặt": [
                    det_plan["stock"][0][i - 1] if i > 0 else float(current_stock)
                    for i in orders_idx
                ],
            }
        )
        st.dataframe(schedule, hide_index=True, width="stretch")
    else:
        st.info("Không cần đặt hàng bổ sung trong kỳ dự báo (theo dự báo trung bình).")

    if order_cost > 0 or holding_cost > 0:
        order_total = orders_idx.size * order_cost
        holding_total = float(det_plan["stock"][0].mean()) * holding_cost / 365 * horizon
        cost_cols = st.columns(3)
        cost_cols[0].metric("Chi phí đặt hàng trong kỳ", f"{order_total:,.0f}")
        cost_cols[1].metric("Chi phí lưu kho trong kỳ", f"{holding_total:,.0f}")
        cost_cols[2].metric("Tổng chi phí tồn kho", f"{order_total + holding_total:,.0f}")

    st.subheader("Chi tiết theo ngày")
    day_status = np.select(
        [det_plan["short"][0] > EPS, det_plan["stock"][0] < ss, det_plan["stock"][0] < rop],
        ["Thiếu hàng", "Dưới tồn an toàn", "Dưới ROP"],
        default="An toàn",
    )
    detail = pd.DataFrame(
        {
            "Ngày": dates,
            "Nhu cầu dự báo": demand_fc.round(1),
            "Hàng về": det_plan["received"][0].round(0),
            "Đặt hàng": det_plan["orders"][0].round(0),
            "Tồn cuối ngày (có bổ sung)": det_plan["stock"][0].round(0),
            "Tồn cuối ngày (không bổ sung)": det_none["stock"][0].round(0),
            "Thiếu hụt": det_plan["short"][0].round(1),
            "P(thiếu hàng) %": ((mc_plan["short"] > EPS).mean(axis=0) * 100).round(1),
            "Trạng thái": day_status,
        }
    )
    st.dataframe(detail, hide_index=True, width="stretch")
    st.download_button(
        "Tải kế hoạch tồn kho (CSV)",
        detail.to_csv(index=False).encode("utf-8-sig"),
        "ke_hoach_ton_kho.csv",
        "text/csv",
    )

# ------------------------------------------------------------------ ABC
st.divider()
st.header("3. Phân loại ABC nhóm hàng")
if abc.empty:
    st.info("Chưa có dữ liệu bán hàng để phân loại ABC.")
else:
    scope_text = f"cửa hàng {store_scope}" if store_scope != ALL_STORE else "toàn bộ cửa hàng"
    counts = abc["abc_class"].value_counts()
    st.caption(
        f"Tính trên {scope_text}: "
        + " · ".join(f"hạng {k}: {int(counts.get(k, 0))} nhóm" for k in "ABC")
        + ". A đóng góp khoảng 80% doanh số, B đến 95%, C phần còn lại."
    )
    colors = {"A": "#dc2626", "B": "#f97316", "C": "#16a34a"}
    abc_chart = px.bar(
        abc, x="family", y="sales", color="abc_class", color_discrete_map=colors,
        title="Doanh số và phân loại ABC",
        labels={"family": "Nhóm hàng", "sales": "Số lượng bán", "abc_class": "Hạng"},
    )
    abc_chart.update_layout(height=420)
    st.plotly_chart(abc_chart, width="stretch")
    with st.expander("Bảng phân loại ABC đầy đủ"):
        st.dataframe(
            abc.rename(columns={"family": "Nhóm hàng", "sales": "Tổng bán",
                                "cum_pct": "Lũy kế (%)", "abc_class": "Hạng"}),
            hide_index=True, width="stretch",
        )
