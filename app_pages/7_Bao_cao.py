# -*- coding: utf-8 -*-
"""Báo cáo điều hành theo tuần / tháng / quý / năm.

Cấu trúc báo cáo:
    Thông tin chung
    I.   Tóm tắt điều hành
    II.  Kết quả kinh doanh và so sánh kỳ
    III. Phân tích cơ cấu (nhóm hàng, cửa hàng, khuyến mãi)
    IV.  Phân loại ABC
    V.   Dự báo và kế hoạch tồn kho
    VI.  Nhận xét và kiến nghị
    Phụ lục: bảng số liệu các kỳ
Xuất được Excel (nhiều sheet) và HTML (mở ra rồi in/lưu PDF).
"""

from __future__ import annotations

import html
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import auth
from app_common import (
    configure_page,
    filter_controls,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from chatbot import build_context, render_report_chat
from inventory import abc_classification

# Tuần tính từ thứ Hai đến Chủ nhật (khớp tuần ISO).
PERIODS = {
    "Tuần": {
        "freq": "W-SUN", "yoy_lag": 52, "trend_n": 13,
        "prev": "tuần trước", "yoy": "cùng tuần năm trước",
    },
    "Tháng": {
        "freq": "M", "yoy_lag": 12, "trend_n": 12,
        "prev": "tháng trước", "yoy": "cùng tháng năm trước",
    },
    "Quý": {
        "freq": "Q", "yoy_lag": 4, "trend_n": 8,
        "prev": "quý trước", "yoy": "cùng quý năm trước",
    },
    "Năm": {
        "freq": "Y", "yoy_lag": 1, "trend_n": 5,
        "prev": "năm trước", "yoy": "năm trước",
    },
}
ALL_STORES = "Tất cả cửa hàng"
ALL_FAMILIES = "Tất cả nhóm hàng"


# ------------------------------------------------------------------ định dạng
def fmt_num(value, digits: int = 0) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def fmt_pct(value, digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:+,.{digits}f}%"


def pct_change(current, base) -> float:
    if base is None or current is None or pd.isna(base) or pd.isna(current) or base <= 0:
        return float("nan")
    return float((current - base) / base * 100)


def pct_series(current: pd.Series, base: pd.Series) -> pd.Series:
    base = base.where(base > 0)
    return (current - base) / base * 100


def trend_word(value: float) -> str:
    return "tăng" if value >= 0 else "giảm"


def period_label(period: pd.Period, key: str) -> str:
    if key == "Tuần":
        iso = period.start_time.isocalendar()
        return f"Tuần {int(iso.week):02d}/{int(iso.year)}"
    if key == "Tháng":
        return f"Tháng {period.month:02d}/{period.year}"
    if key == "Quý":
        return f"Quý {period.quarter}/{period.year}"
    return f"Năm {period.year}"


def date_range_text(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%d/%m/%Y} – {end:%d/%m/%Y}"


# ------------------------------------------------------------------ tính toán
def build_period_table(daily: pd.Series, key: str) -> pd.DataFrame:
    """Gộp doanh số ngày theo kỳ, kèm độ phủ dữ liệu và tăng trưởng."""
    cfg = PERIODS[key]
    freq = cfg["freq"]
    data_min, data_max = daily.index.min(), daily.index.max()

    grouped = daily.groupby(daily.index.to_period(freq)).agg(["sum", "count"])
    grouped.columns = ["sales", "days_with_data"]
    table = grouped.reindex(
        pd.period_range(grouped.index.min(), grouped.index.max(), freq=freq)
    )
    table["sales"] = table["sales"].fillna(0.0)
    table["days_with_data"] = table["days_with_data"].fillna(0).astype(int)

    start = table.index.start_time.normalize()
    end = table.index.end_time.normalize()
    covered_start = start.where(start >= data_min, data_min)
    covered_end = end.where(end <= data_max, data_max)
    table["start"] = start
    table["end"] = end
    table["days_in_period"] = (end - start).days + 1
    table["days_covered"] = (covered_end - covered_start).days + 1
    table["full"] = (start >= data_min) & (end <= data_max)
    table["avg_day"] = table["sales"] / table["days_covered"].clip(lower=1)

    full = table["full"]
    lag = cfg["yoy_lag"]
    # Tổng lượng bán chỉ so sánh khi cả hai kỳ đều đủ dữ liệu; TB/ngày luôn so sánh được.
    table["growth_total"] = pct_series(table["sales"], table["sales"].shift(1)).where(
        full & full.shift(1, fill_value=False)
    )
    table["growth_avg"] = pct_series(table["avg_day"], table["avg_day"].shift(1))
    table["yoy_total"] = pct_series(table["sales"], table["sales"].shift(lag)).where(
        full & full.shift(lag, fill_value=False)
    )
    table["yoy_avg"] = pct_series(table["avg_day"], table["avg_day"].shift(lag))
    return table


def slice_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["date"] >= start) & (frame["date"] < end + pd.Timedelta(days=1))]


def compare_dimension(cur: pd.DataFrame, prev: pd.DataFrame, column: str, label: str):
    """Doanh số kỳ này và kỳ trước theo một chiều (nhóm hàng, cửa hàng)."""
    if column not in cur.columns or cur.empty:
        return None
    current = cur.groupby(cur[column].astype(str))["sales"].sum()
    previous = (
        prev.groupby(prev[column].astype(str))["sales"].sum()
        if not prev.empty
        else pd.Series(dtype=float)
    )
    result = pd.concat([current.rename("cur"), previous.rename("prev")], axis=1).fillna(0.0)
    result = result.sort_values("cur", ascending=False)
    total = float(result["cur"].sum())
    result["share"] = result["cur"] / total * 100 if total > 0 else 0.0
    result["change"] = result["cur"] - result["prev"]
    result["change_pct"] = pct_series(result["cur"], result["prev"])
    result.index.name = label
    result = result.reset_index()
    return result[[label, "cur", "prev", "change", "change_pct", "share"]].rename(
        columns={
            "cur": "Kỳ này",
            "prev": "Kỳ trước",
            "change": "Chênh lệch",
            "change_pct": "+/- so kỳ trước (%)",
            "share": "Tỷ trọng (%)",
        }
    )


def period_overview(table: pd.DataFrame, key: str) -> pd.DataFrame:
    view = pd.DataFrame(
        {
            "Kỳ": [period_label(p, key) for p in table.index],
            "Từ ngày": table["start"].to_numpy(),
            "Đến ngày": table["end"].to_numpy(),
            "Lượng bán": table["sales"].to_numpy(),
            "TB/ngày": table["avg_day"].to_numpy(),
            "+/- kỳ trước: tổng (%)": table["growth_total"].to_numpy(),
            "+/- kỳ trước: TB/ngày (%)": table["growth_avg"].to_numpy(),
            "+/- cùng kỳ NT: tổng (%)": table["yoy_total"].to_numpy(),
            "+/- cùng kỳ NT: TB/ngày (%)": table["yoy_avg"].to_numpy(),
            "Đủ kỳ": table["full"].to_numpy(),
        }
    )
    return view.iloc[::-1].reset_index(drop=True)


# ------------------------------------------------------------------ hiển thị / xuất
def show_table(df: pd.DataFrame) -> None:
    config = {}
    for column in df.columns:
        if pd.api.types.is_float_dtype(df[column]):
            config[column] = st.column_config.NumberColumn(format="%.1f")
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            config[column] = st.column_config.DateColumn(format="DD/MM/YYYY")
    st.dataframe(df, column_config=config, width="stretch", hide_index=True)


def table_to_html(df: pd.DataFrame) -> str:
    frame = df.copy()
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            frame[column] = series.map({True: "Có", False: "Không"})
        elif pd.api.types.is_integer_dtype(series):
            frame[column] = series.map(lambda v: f"{v:,}")
        elif pd.api.types.is_float_dtype(series):
            frame[column] = series.map(lambda v: "—" if pd.isna(v) else f"{v:,.1f}")
        elif pd.api.types.is_datetime64_any_dtype(series):
            frame[column] = series.dt.strftime("%d/%m/%Y")
    return frame.to_html(index=False, escape=True, border=0, classes="tbl")


def build_excel(sheets: dict) -> bytes | None:
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine=engine) as writer:
                for name, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=name[:31], index=False)
                    if engine == "openpyxl":
                        sheet = writer.sheets[name[:31]]
                        for idx, column in enumerate(frame.columns, start=1):
                            width = max(
                                [len(str(column))]
                                + [len(str(v)) for v in frame[column].head(200)]
                            )
                            sheet.column_dimensions[
                                sheet.cell(row=1, column=idx).column_letter
                            ].width = min(max(width + 2, 10), 40)
            return buffer.getvalue()
        except ImportError:
            continue
    return None


def build_html_report(title, meta, summary, kpis, sections, recommendations) -> str:
    esc = html.escape
    meta_rows = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in meta
    )
    kpi_cells = "".join(
        f'<div class="kpi"><span>{esc(label)}</span><b>{esc(value)}</b>'
        f"<em>{esc(note)}</em></div>"
        for label, value, note in kpis
    )
    summary_html = "".join(f"<li>{esc(line)}</li>" for line in summary)
    recs_html = "".join(f"<li>{esc(line)}</li>" for line in recommendations)
    body = ""
    for heading, content in sections:
        body += f"<h2>{esc(heading)}</h2>"
        if isinstance(content, pd.DataFrame):
            body += table_to_html(content) if not content.empty else "<p>Không có dữ liệu.</p>"
        else:
            body += f"<p>{esc(str(content))}</p>"
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#0f172a;max-width:980px;margin:24px auto;padding:0 16px;font-size:14px}}
header{{border-bottom:3px solid #0d5c24;padding-bottom:8px;margin-bottom:16px}}
h1{{font-size:22px;margin:0;color:#0d5c24;text-transform:uppercase}}
h2{{font-size:16px;margin:24px 0 8px;color:#0d5c24;border-left:5px solid #16a34a;padding-left:8px}}
table{{border-collapse:collapse;width:100%}}
.meta th{{text-align:left;width:200px;padding:4px 8px;background:#f1f5f9}}
.meta td{{padding:4px 8px;border-bottom:1px solid #e2e8f0}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0}}
.kpi{{border:1px solid #cbd5e1;border-radius:8px;padding:10px}}
.kpi span{{display:block;font-size:12px;color:#64748b}}
.kpi b{{display:block;font-size:20px;margin:4px 0}}
.kpi em{{font-style:normal;font-size:12px;color:#475569}}
.tbl th{{background:#0d5c24;color:#fff;text-align:left;padding:6px;font-size:12px}}
.tbl td{{padding:5px 6px;border-bottom:1px solid #e2e8f0;font-size:12px}}
.sign{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;text-align:center;margin-top:48px}}
.sign div{{height:90px}}
.note{{color:#64748b;font-size:12px;margin-top:24px}}
@media print{{body{{margin:0}} h2{{break-after:avoid}} table{{break-inside:auto}} tr{{break-inside:avoid}}}}
</style></head><body>
<header><h1>{esc(title)}</h1></header>
<table class="meta">{meta_rows}</table>
<h2>I. Tóm tắt điều hành</h2><div class="kpis">{kpi_cells}</div><ul>{summary_html}</ul>
{body}
<h2>VI. Nhận xét và kiến nghị</h2><ul>{recs_html}</ul>
<div class="sign"><div><b>Người lập</b><br><i>(Ký, ghi rõ họ tên)</i></div>
<div><b>Người kiểm tra</b><br><i>(Ký, ghi rõ họ tên)</i></div>
<div><b>Người phê duyệt</b><br><i>(Ký, ghi rõ họ tên)</i></div></div>
<p class="note">Nhận xét được sinh tự động từ số liệu; người lập báo cáo cần kiểm tra trước khi ban hành.</p>
</body></html>"""


def trend_figure(table: pd.DataFrame, selected: pd.Period, key: str) -> go.Figure:
    count = PERIODS[key]["trend_n"]
    position = table.index.get_loc(selected)
    view = table.iloc[max(0, position - count + 1) : position + 1]
    labels = [
        period_label(p, key) + ("" if view.at[p, "full"] else "*") for p in view.index
    ]
    colors = ["#0d5c24" if p == selected else "#94a3b8" for p in view.index]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(x=labels, y=view["sales"], marker_color=colors, name="Lượng bán")
    figure.add_scatter(
        x=labels,
        y=view["growth_avg"],
        mode="lines+markers",
        name="+/- TB/ngày so kỳ trước (%)",
        line={"color": "#f59e0b"},
        secondary_y=True,
    )
    figure.update_layout(title=f"Xu hướng {len(view)} {key.lower()} gần nhất", hovermode="x unified")
    figure.update_yaxes(title_text="Lượng bán", secondary_y=False)
    figure.update_yaxes(title_text="Tăng trưởng (%)", secondary_y=True, showgrid=False)
    return figure


# ================================================================== TRANG
configure_page("Báo cáo tổng hợp")
initialize_session_state()

st.title("Báo cáo tổng hợp")
st.caption(
    "Báo cáo điều hành theo tuần, tháng, quý hoặc năm: kết quả kinh doanh, so sánh kỳ, "
    "cơ cấu, phân loại ABC, dự báo và tồn kho."
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()
if "date" not in sales or "sales" not in sales:
    st.error("Dữ liệu cần có cột date và sales để lập báo cáo.")
    st.stop()

filtered = filter_controls(sales, key_prefix="report")
filtered = filtered.assign(
    date=pd.to_datetime(filtered["date"], errors="coerce"),
    sales=pd.to_numeric(filtered["sales"], errors="coerce").fillna(0.0),
).dropna(subset=["date"])
if filtered.empty:
    st.warning("Không có dữ liệu cho bộ lọc đã chọn.")
    st.stop()

daily = filtered.groupby(filtered["date"].dt.normalize())["sales"].sum().sort_index()
daily.index = pd.DatetimeIndex(daily.index)

control_type, control_period = st.columns([1, 2])
period_key = control_type.radio(
    "Loại báo cáo", list(PERIODS), horizontal=True, key="report_period_type"
)
cfg = PERIODS[period_key]
table = build_period_table(daily, period_key)
options = list(table.index[::-1])
default_index = next((i for i, p in enumerate(options) if bool(table.at[p, "full"])), 0)


def option_label(period: pd.Period) -> str:
    text = (
        f"{period_label(period, period_key)} "
        f"({date_range_text(table.at[period, 'start'], table.at[period, 'end'])})"
    )
    return text if table.at[period, "full"] else text + " · chưa đủ kỳ"


selected = control_period.selectbox(
    "Kỳ báo cáo",
    options,
    index=default_index,
    format_func=option_label,
    key=f"report_period_{period_key}",
)

# ---- số liệu kỳ được chọn
start, end = table.at[selected, "start"], table.at[selected, "end"]
total = float(table.at[selected, "sales"])
avg_day = float(table.at[selected, "avg_day"])
days_with_data = int(table.at[selected, "days_with_data"])
days_in_period = int(table.at[selected, "days_in_period"])
days_covered = int(table.at[selected, "days_covered"])
is_full = bool(table.at[selected, "full"])
growth_total = table.at[selected, "growth_total"]
growth_avg = table.at[selected, "growth_avg"]
yoy_total = table.at[selected, "yoy_total"]
yoy_avg = table.at[selected, "yoy_avg"]
label = period_label(selected, period_key)

has_prev = (selected - 1) in table.index
current = slice_period(filtered, start, end)
previous = (
    slice_period(filtered, table.at[selected - 1, "start"], table.at[selected - 1, "end"])
    if has_prev
    else filtered.iloc[0:0]
)

family_table = compare_dimension(current, previous, "family", "Nhóm hàng")
store_table = compare_dimension(current, previous, "store_nbr", "Cửa hàng")
daily_cur = current.groupby(current["date"].dt.normalize())["sales"].sum()
peak_day = daily_cur.idxmax() if not daily_cur.empty else None
peak_value = float(daily_cur.max()) if not daily_cur.empty else float("nan")

abc = None
if "family" in current.columns and not current.empty:
    abc = abc_classification(current)

# ---- dự báo và tồn kho (lấy từ session, phạm vi theo lúc huấn luyện)
forecast = st.session_state.forecast_result
inventory_plan = st.session_state.inventory_plan
metrics = st.session_state.test_metrics
training_scope = st.session_state.get("training_filter") or {}
scope_store = training_scope.get("store_nbr", ALL_STORES)
scope_family = training_scope.get("family", ALL_FAMILIES)
scope_match = (
    scope_store == st.session_state.selected_store
    and scope_family == st.session_state.selected_family
)

forecast_by_period = None
forecast_avg = float("nan")
if forecast is not None and not forecast.empty:
    fc = forecast.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    forecast_avg = float(fc["forecast"].mean())
    grouped = fc.groupby(fc["date"].dt.to_period(cfg["freq"]))["forecast"].agg(["sum", "size"])
    rows = []
    for period, record in grouped.iterrows():
        full_days = (period.end_time.normalize() - period.start_time.normalize()).days + 1
        rows.append(
            {
                "Kỳ": period_label(period, period_key),
                "Lượng bán dự báo": float(record["sum"]),
                "Số ngày dự báo": int(record["size"]),
                "TB/ngày": float(record["sum"] / record["size"]),
                "Đủ kỳ": int(record["size"]) == full_days,
            }
        )
    forecast_by_period = pd.DataFrame(rows)
forecast_vs_current = pct_change(forecast_avg, avg_day) if scope_match else float("nan")

# ---- khuyến mãi
promo_summary = None
if "onpromotion" in current.columns and not current.empty:
    flag = pd.to_numeric(current["onpromotion"], errors="coerce").fillna(0) > 0
    if flag.any() and (~flag).any():
        with_promo = float(current.loc[flag, "sales"].mean())
        without_promo = float(current.loc[~flag, "sales"].mean())
        promo_summary = pd.DataFrame(
            {
                "Nhóm": ["Có khuyến mãi", "Không khuyến mãi"],
                "Số dòng": [int(flag.sum()), int((~flag).sum())],
                "Lượng bán TB/dòng": [with_promo, without_promo],
            }
        )
        promo_uplift = pct_change(with_promo, without_promo)
    else:
        promo_uplift = float("nan")
else:
    promo_uplift = float("nan")

# ---- nhận xét tự động
top_family = family_table.iloc[0] if family_table is not None and not family_table.empty else None
top_store = store_table.iloc[0] if store_table is not None and not store_table.empty else None
top3_share = (
    float(family_table["Tỷ trọng (%)"].head(3).sum()) if family_table is not None else float("nan")
)

summary = [
    f"{label} ({date_range_text(start, end)}): tổng lượng bán đạt {fmt_num(total)} đơn vị, "
    f"trung bình {fmt_num(avg_day, 1)} đơn vị/ngày."
]
if has_prev and pd.notna(growth_avg):
    line = f"So với {cfg['prev']}: lượng bán TB/ngày {trend_word(growth_avg)} {abs(growth_avg):.1f}%"
    if pd.notna(growth_total):
        line += f", tổng lượng bán {trend_word(growth_total)} {abs(growth_total):.1f}%"
    summary.append(line + ".")
else:
    summary.append(f"Chưa có {cfg['prev']} để so sánh.")
if pd.notna(yoy_avg):
    summary.append(
        f"So với {cfg['yoy']}: lượng bán TB/ngày {trend_word(yoy_avg)} {abs(yoy_avg):.1f}%."
    )
else:
    summary.append(f"Chưa đủ dữ liệu để so sánh với {cfg['yoy']}.")
if top_family is not None:
    summary.append(
        f"Nhóm hàng dẫn đầu là {top_family['Nhóm hàng']} "
        f"({top_family['Tỷ trọng (%)']:.1f}% tổng lượng bán); 3 nhóm đầu chiếm {top3_share:.1f}%."
    )
if peak_day is not None:
    summary.append(f"Ngày cao điểm: {peak_day:%d/%m/%Y} với {fmt_num(peak_value)} đơn vị.")
if not is_full:
    summary.append(
        f"Lưu ý: kỳ báo cáo chưa đủ dữ liệu ({days_covered}/{days_in_period} ngày), "
        "nên cần thận trọng khi so sánh tổng lượng bán."
    )

recommendations = []
if has_prev and pd.notna(growth_avg) and growth_avg <= -10:
    decliners = (
        family_table[family_table["Kỳ trước"] > 0].sort_values("Chênh lệch").head(1)
        if family_table is not None
        else None
    )
    detail = (
        f" Nhóm hàng giảm mạnh nhất: {decliners.iloc[0]['Nhóm hàng']}."
        if decliners is not None and not decliners.empty and decliners.iloc[0]["Chênh lệch"] < 0
        else ""
    )
    recommendations.append(
        f"Lượng bán TB/ngày giảm {abs(growth_avg):.1f}% so với {cfg['prev']}: "
        f"cần rà soát khuyến mãi, tồn kho và nguyên nhân giảm.{detail}"
    )
if has_prev and pd.notna(growth_avg) and growth_avg >= 10:
    recommendations.append(
        f"Lượng bán TB/ngày tăng {growth_avg:.1f}% so với {cfg['prev']}: "
        "kiểm tra khả năng đáp ứng tồn kho và lịch nhập hàng."
    )
if pd.notna(top3_share) and top3_share >= 60:
    recommendations.append(
        f"Doanh số tập trung cao (3 nhóm đầu chiếm {top3_share:.1f}%): "
        "cần theo dõi sát tồn kho và chuỗi cung ứng của các nhóm này."
    )
if abc is not None:
    class_a = int((abc["abc_class"] == "A").sum())
    recommendations.append(
        f"Có {class_a} nhóm hàng loại A (đóng góp khoảng 80% lượng bán): "
        "ưu tiên mức phục vụ cao và kiểm soát tồn kho chặt."
    )
if inventory_plan and inventory_plan.get("need_reorder"):
    recommendations.append(
        f"Tồn kho nhập tay đang ở hoặc dưới điểm đặt hàng lại "
        f"({fmt_num(inventory_plan['reorder_point'], 1)}): đề xuất đặt "
        f"{fmt_num(inventory_plan['suggested_order_qty'], 1)} đơn vị."
    )
if pd.notna(forecast_vs_current) and abs(forecast_vs_current) >= 10:
    recommendations.append(
        f"Dự báo TB/ngày {trend_word(forecast_vs_current)} {abs(forecast_vs_current):.1f}% "
        f"so với {label}: điều chỉnh kế hoạch nhập hàng cho phù hợp."
    )
if not recommendations:
    recommendations.append("Chưa phát hiện biến động bất thường; duy trì kế hoạch hiện tại.")

# ---- thông tin chung
try:
    me = auth.current_user()
except RuntimeError:
    me = None
author = (me or {}).get("display_name") or (me or {}).get("username") or "—"
data_state = "đã làm sạch" if st.session_state.processed_df is not None else "dữ liệu gốc"
meta = [
    ("Loại báo cáo", f"Báo cáo {period_key.lower()}"),
    ("Kỳ báo cáo", f"{label} ({date_range_text(start, end)})"),
    (
        "Phạm vi",
        f"{st.session_state.selected_store} · {st.session_state.selected_family}",
    ),
    ("Nguồn dữ liệu", f"{st.session_state.source_label or 'Dữ liệu đã nạp'} ({data_state})"),
    ("Ngày lập báo cáo", f"{datetime.now():%d/%m/%Y %H:%M}"),
    ("Người lập", author),
]

# ================================================================== HIỂN THỊ
st.divider()
st.subheader("Thông tin chung")
with st.container(border=True):
    for name, value in meta:
        st.markdown(f"**{name}:** {value}")

# ---- I. Tóm tắt điều hành
st.subheader("I. Tóm tắt điều hành")
kpis = [
    (
        "Tổng lượng bán",
        fmt_num(total),
        f"{fmt_pct(growth_total)} so với {cfg['prev']}" if pd.notna(growth_total) else "Chưa so sánh được tổng",
    ),
    ("Lượng bán TB/ngày", fmt_num(avg_day, 1), f"{fmt_pct(growth_avg)} so với {cfg['prev']}"),
    ("TB/ngày so " + cfg["yoy"], fmt_pct(yoy_avg) if pd.notna(yoy_avg) else "Chưa đủ dữ liệu", ""),
    ("Số ngày có dữ liệu", f"{days_with_data}/{days_in_period}", "Đủ kỳ" if is_full else "Kỳ chưa đủ"),
]
cols = st.columns(4)
cols[0].metric("Tổng lượng bán", fmt_num(total), delta=fmt_pct(growth_total) if pd.notna(growth_total) else None,
               help=f"So với {cfg['prev']}; chỉ hiện khi cả hai kỳ đều đủ dữ liệu.")
cols[1].metric("Lượng bán TB/ngày", fmt_num(avg_day, 1), delta=fmt_pct(growth_avg) if pd.notna(growth_avg) else None,
               help=f"So với {cfg['prev']}.")
cols[2].metric(f"TB/ngày so {cfg['yoy']}", fmt_pct(yoy_avg) if pd.notna(yoy_avg) else "Chưa đủ dữ liệu")
cols[3].metric("Số ngày có dữ liệu", f"{days_with_data}/{days_in_period}",
               help="Số ngày có phát sinh dữ liệu trên số ngày của kỳ.")
cols2 = st.columns(4)
cols2[0].metric("Ngày cao điểm", f"{peak_day:%d/%m/%Y}" if peak_day is not None else "—",
                help=f"Lượng bán: {fmt_num(peak_value)}")
cols2[1].metric("Nhóm hàng dẫn đầu", str(top_family["Nhóm hàng"]) if top_family is not None else "—",
                help=f"Tỷ trọng {top_family['Tỷ trọng (%)']:.1f}%" if top_family is not None else None)
cols2[2].metric("Cửa hàng dẫn đầu", str(top_store["Cửa hàng"]) if top_store is not None else "—",
                help=f"Tỷ trọng {top_store['Tỷ trọng (%)']:.1f}%" if top_store is not None else None)
cols2[3].metric("Số nhóm hàng loại A", f"{int((abc['abc_class'] == 'A').sum())}" if abc is not None else "—")
for line in summary:
    st.markdown(f"- {line}")
if not is_full:
    st.warning(
        f"{label} chưa đủ dữ liệu ({days_covered}/{days_in_period} ngày). "
        "Hãy ưu tiên so sánh theo TB/ngày."
    )

# ---- II. Kết quả kinh doanh
st.subheader("II. Kết quả kinh doanh và so sánh kỳ")
st.plotly_chart(trend_figure(table, selected, period_key), theme=None, width="stretch")
st.caption("Cột đậm là kỳ báo cáo. Dấu * là kỳ chưa đủ dữ liệu.")

if not daily_cur.empty:
    day_index = pd.date_range(
        max(start, daily.index.min()), min(end, daily.index.max()), freq="D"
    )
    daily_view = daily_cur.reindex(day_index, fill_value=0.0)
    day_figure = go.Figure()
    day_figure.add_scatter(x=daily_view.index, y=daily_view.values, mode="lines", name="Lượng bán")
    day_figure.add_scatter(
        x=daily_view.index,
        y=daily_view.rolling(7, min_periods=1).mean().values,
        mode="lines",
        name="Trung bình trượt 7 ngày",
        line={"dash": "dash"},
    )
    day_figure.update_layout(title=f"Diễn biến theo ngày trong {label.lower()}", hovermode="x unified")
    st.plotly_chart(day_figure, theme=None, width="stretch")
    top_days = (
        daily_cur.nlargest(5)
        .rename("Lượng bán")
        .rename_axis("Ngày")
        .reset_index()
    )
    top_days["Ngày"] = pd.to_datetime(top_days["Ngày"])
    st.markdown("**Top 5 ngày bán cao nhất**")
    show_table(top_days)

# ---- III. Cơ cấu
st.subheader("III. Phân tích cơ cấu")
if family_table is not None:
    st.markdown("**Theo nhóm hàng (top 15)**")
    show_table(family_table.head(15))
if store_table is not None:
    st.markdown("**Theo cửa hàng (top 10)**")
    show_table(store_table.head(10))
if family_table is None and store_table is None:
    st.info("Dữ liệu không có cột nhóm hàng hoặc cửa hàng.")
if promo_summary is not None:
    st.markdown("**Khuyến mãi**")
    show_table(promo_summary)
    st.caption(
        f"Chênh lệch lượng bán TB/dòng giữa dòng có và không khuyến mãi: {fmt_pct(promo_uplift)}. "
        "Chỉ mang tính tham khảo, chưa loại trừ nhóm hàng, cửa hàng và mùa vụ."
    )

# ---- IV. ABC
st.subheader("IV. Phân loại ABC")
if abc is None:
    st.info("Không có dữ liệu nhóm hàng để phân loại ABC.")
else:
    abc_view = abc.rename(
        columns={
            "family": "Nhóm hàng",
            "sales": "Lượng bán",
            "cum_pct": "Tỷ trọng tích lũy (%)",
            "abc_class": "Loại ABC",
        }
    )
    counts = abc["abc_class"].value_counts()
    st.caption(
        f"Phân loại trên dữ liệu của {label.lower()}: "
        f"A = {int(counts.get('A', 0))}, B = {int(counts.get('B', 0))}, C = {int(counts.get('C', 0))} nhóm hàng."
    )
    show_table(abc_view)

# ---- V. Dự báo và tồn kho
st.subheader("V. Dự báo và kế hoạch tồn kho")
scope_text = f"{scope_store} · {scope_family}"
if forecast_by_period is None:
    st.info("Chưa có dự báo. Tạo dự báo ở trang **Dự báo doanh số**.")
else:
    st.caption(f"Phạm vi mô hình và dự báo: {scope_text}.")
    forecast_cols = st.columns(3)
    forecast_cols[0].metric("Tổng lượng dự báo", fmt_num(float(forecast["forecast"].sum()), 1))
    forecast_cols[1].metric("Dự báo TB/ngày", fmt_num(forecast_avg, 1))
    forecast_cols[2].metric(
        f"So với TB/ngày {label}",
        fmt_pct(forecast_vs_current) if pd.notna(forecast_vs_current) else "Không so sánh",
    )
    if not scope_match:
        st.warning(
            "Phạm vi báo cáo khác phạm vi lúc huấn luyện mô hình nên không so sánh trực tiếp "
            "dự báo với kỳ báo cáo."
        )
    st.markdown(f"**Dự báo theo {period_key.lower()}**")
    show_table(forecast_by_period)
if metrics:
    st.markdown("**Độ chính xác mô hình (tập kiểm định)**")
    metric_cols = st.columns(4)
    for column, name in zip(metric_cols, ("MAE", "RMSE", "MAPE (%)", "WAPE (%)")):
        value = metrics.get(name)
        column.metric(name, f"{value:,.2f}" if value is not None and pd.notna(value) else "Không xác định")
    st.caption(f"Cấu hình: {st.session_state.model_config}")
else:
    st.info("Chưa có kết quả huấn luyện mô hình.")

inventory_table = None
if inventory_plan:
    st.markdown("**Kế hoạch tồn kho**")
    inv_cols = st.columns(4)
    inv_cols[0].metric("Tồn kho an toàn", fmt_num(inventory_plan["safety_stock"], 1))
    inv_cols[1].metric("Điểm đặt hàng lại (ROP)", fmt_num(inventory_plan["reorder_point"], 1))
    inv_cols[2].metric("EOQ", fmt_num(inventory_plan["eoq"], 1) if inventory_plan["eoq"] is not None else "Chưa nhập chi phí")
    inv_cols[3].metric(
        "Trạng thái",
        "Cần đặt hàng" if inventory_plan["need_reorder"] else "Chưa cần đặt hàng",
        help=f"Lượng đề xuất đặt: {fmt_num(inventory_plan['suggested_order_qty'], 1)}",
    )
    st.caption("Tồn kho hiện tại do người dùng nhập tay ở trang Quản trị tồn kho.")
    days_supply = inventory_plan["days_of_supply"]
    inventory_table = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Nhu cầu TB/ngày", "Tồn kho an toàn", "Điểm đặt hàng lại (ROP)", "EOQ",
                "Cần đặt hàng", "Lượng đề xuất đặt", "Số ngày cung ứng",
            ],
            "Giá trị": [
                fmt_num(inventory_plan["avg_daily_demand"], 1),
                fmt_num(inventory_plan["safety_stock"], 1),
                fmt_num(inventory_plan["reorder_point"], 1),
                fmt_num(inventory_plan["eoq"], 1) if inventory_plan["eoq"] is not None else "—",
                "Có" if inventory_plan["need_reorder"] else "Không",
                fmt_num(inventory_plan["suggested_order_qty"], 1),
                fmt_num(days_supply, 1) if days_supply != float("inf") else "Không giới hạn",
            ],
        }
    )
else:
    st.info("Chưa có kế hoạch tồn kho. Thiết lập ở trang **Quản trị tồn kho**.")

# ---- VI. Nhận xét
st.subheader("VI. Nhận xét và kiến nghị")
for line in recommendations:
    st.markdown(f"- {line}")
st.caption("Nhận xét được sinh tự động từ số liệu; người lập báo cáo cần kiểm tra trước khi ban hành.")

# ---- Phụ lục
overview = period_overview(table, period_key)
with st.expander(f"Phụ lục: số liệu tất cả các {period_key.lower()}"):
    show_table(overview)

# ---- Xuất báo cáo
st.divider()
st.subheader("Xuất báo cáo")
STUB_NAMES = {"Tuần": "tuan", "Tháng": "thang", "Quý": "quy", "Năm": "nam"}
file_stub = f"bao_cao_{STUB_NAMES[period_key]}_{start:%Y%m%d}_{end:%Y%m%d}"

html_sections = [("II. Kết quả kinh doanh và so sánh kỳ", overview.head(cfg["trend_n"]))]
if family_table is not None:
    html_sections.append(("III. Cơ cấu theo nhóm hàng (top 15)", family_table.head(15)))
if store_table is not None:
    html_sections.append(("III. Cơ cấu theo cửa hàng (top 10)", store_table.head(10)))
if abc is not None:
    html_sections.append(("IV. Phân loại ABC", abc_view))
if forecast_by_period is not None:
    html_sections.append((f"V. Dự báo theo {period_key.lower()} ({scope_text})", forecast_by_period))
if inventory_table is not None:
    html_sections.append(("V. Kế hoạch tồn kho", inventory_table))
html_report = build_html_report(
    f"Báo cáo {period_key.lower()} - {label}", meta, summary, kpis, html_sections, recommendations
)

sheets = {"Thong_tin": pd.DataFrame(meta, columns=["Chỉ tiêu", "Giá trị"])}
sheets["Tom_tat"] = pd.DataFrame({"Nội dung": summary + ["—"] + recommendations})
sheets["So_sanh_ky"] = overview
if family_table is not None:
    sheets["Nhom_hang"] = family_table
if store_table is not None:
    sheets["Cua_hang"] = store_table
if abc is not None:
    sheets["ABC"] = abc_view
if forecast_by_period is not None:
    sheets["Du_bao_theo_ky"] = forecast_by_period
if inventory_table is not None:
    sheets["Ton_kho"] = inventory_table

export_cols = st.columns(3)
export_cols[0].download_button(
    "Tải báo cáo HTML (in / lưu PDF)",
    html_report.encode("utf-8"),
    f"{file_stub}.html",
    "text/html",
)
excel_bytes = build_excel(sheets)
if excel_bytes is not None:
    export_cols[1].download_button(
        "Tải báo cáo Excel",
        excel_bytes,
        f"{file_stub}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    export_cols[1].info("Cài `openpyxl` để xuất Excel: `pip install openpyxl`.")
export_cols[2].download_button(
    "Tải bảng so sánh các kỳ (CSV)",
    overview.to_csv(index=False).encode("utf-8-sig"),
    f"{file_stub}_so_sanh_ky.csv",
    "text/csv",
)
if forecast is not None and not forecast.empty:
    st.download_button(
        "Tải forecast.csv",
        forecast.to_csv(index=False).encode("utf-8-sig"),
        "forecast.csv",
        "text/csv",
    )

# ---- Trợ lý AI hỏi đáp (dùng đúng số liệu của báo cáo đang hiển thị)
chat_context = build_context(
    meta=meta,
    summary=summary,
    recommendations=recommendations,
    overview=overview,
    family_table=family_table,
    store_table=store_table,
    abc=abc,
    promo_summary=promo_summary,
    forecast_by_period=forecast_by_period,
    metrics=metrics,
    inventory_plan=inventory_plan,
    scope_match=scope_match,
)
render_report_chat(
    chat_context,
    summary,
    recommendations,
    scope_label=f"{label} · {st.session_state.selected_store} · {st.session_state.selected_family}",
)