# -*- coding: utf-8 -*-
"""Dashboard tổng quan: KPI, xu hướng, bản đồ vệ tinh cửa hàng, ABC, dự báo & tồn kho."""

import math
import unicodedata

import numpy as np
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
from inventory import abc_classification
from theme import CHART_COLORS, kpi_cards, page_intro, status_tag
from utils import aggregate_daily

configure_page("Dashboard")
initialize_session_state()

ALL_FAMILY = "Tất cả nhóm hàng"
ALL_STORE = "Tất cả cửa hàng"
ESRI_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_LABELS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
    "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)

# Tọa độ trung tâm các thành phố trong bộ dữ liệu Favorita (Ecuador).
# Dùng khi bảng stores không có sẵn cột latitude/longitude.
CITY_COORDS = {
    "quito": (-0.1807, -78.4678),
    "santo domingo": (-0.2530, -79.1754),
    "cayambe": (0.0417, -78.1450),
    "latacunga": (-0.9352, -78.6155),
    "riobamba": (-1.6636, -78.6546),
    "ibarra": (0.3517, -78.1223),
    "guaranda": (-1.5905, -79.0010),
    "puyo": (-1.4924, -78.0024),
    "ambato": (-1.2491, -78.6168),
    "guayaquil": (-2.1710, -79.9224),
    "salinas": (-2.2146, -80.9535),
    "daule": (-1.8622, -79.9778),
    "babahoyo": (-1.8022, -79.5344),
    "quevedo": (-1.0225, -79.4604),
    "playas": (-2.6317, -80.3886),
    "libertad": (-2.2234, -80.9086),
    "cuenca": (-2.9001, -79.0059),
    "loja": (-3.9931, -79.2042),
    "machala": (-3.2581, -79.9554),
    "esmeraldas": (0.9592, -79.6540),
    "manta": (-0.9677, -80.7089),
    "el carmen": (-0.2667, -79.4333),
}
DOW_LABELS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


# ------------------------------------------------------------------ tiện ích
def _norm(text) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.lower().strip()


def _signature(frame: pd.DataFrame) -> tuple:
    return (
        len(frame),
        str(frame["date"].min()),
        str(frame["date"].max()),
        round(float(frame["sales"].fillna(0).sum()), 2),
    )


def _fmt(value: float) -> str:
    value = float(value)
    for unit, size in (("tỷ", 1e9), ("tr", 1e6), ("nghìn", 1e3)):
        if abs(value) >= size:
            return f"{value / size:,.1f} {unit}"
    return f"{value:,.0f}"


def _find_table(tables: dict, name: str):
    for key, frame in (tables or {}).items():
        if _norm(key) == name:
            return frame
    return None


def build_store_master(sales: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Gom thông tin cửa hàng (city, state, type, cluster, tọa độ) từ bảng stores hoặc bảng bán hàng."""
    aliases = {"lat": "latitude", "lng": "longitude", "lon": "longitude", "long": "longitude"}
    wanted = ["store_nbr", "city", "state", "type", "cluster", "latitude", "longitude"]
    master = _find_table(tables, "stores")
    if master is not None and "store_nbr" in [str(c).lower() for c in master.columns]:
        master = master.copy()
    elif "store_nbr" in sales.columns:
        cols = [c for c in sales.columns if str(c).lower() in wanted + list(aliases)]
        master = sales[cols].drop_duplicates("store_nbr").copy()
    else:
        return pd.DataFrame(columns=wanted)

    master.columns = [aliases.get(str(c).lower(), str(c).lower()) for c in master.columns]
    master = master.loc[:, ~master.columns.duplicated()]
    master["store_nbr"] = master["store_nbr"].astype(str)
    master = master.drop_duplicates("store_nbr")
    for col in ("city", "state", "type", "cluster"):
        if col not in master:
            master[col] = "Không rõ"
        master[col] = master[col].fillna("Không rõ").astype(str)

    if {"latitude", "longitude"} <= set(master.columns):
        lat = pd.to_numeric(master["latitude"], errors="coerce")
        lon = pd.to_numeric(master["longitude"], errors="coerce")
        master["approx"] = False
    else:
        lat = pd.Series(np.nan, index=master.index)
        lon = pd.Series(np.nan, index=master.index)
        master["approx"] = True

    # Điền tọa độ thiếu theo thành phố, rải đều các cửa hàng cùng thành phố để không chồng lên nhau.
    missing = lat.isna() | lon.isna()
    city_key = master["city"].map(_norm)
    for city, idx in master[missing].groupby(city_key[missing]).groups.items():
        if city not in CITY_COORDS:
            continue
        base_lat, base_lon = CITY_COORDS[city]
        for k, row in enumerate(sorted(idx)):
            if len(idx) == 1:
                offset = (0.0, 0.0)
            else:
                angle = k * 2.399963  # góc vàng
                radius = 0.011 * math.sqrt(k + 1)
                offset = (radius * math.sin(angle), radius * math.cos(angle))
            lat.loc[row] = base_lat + offset[0]
            lon.loc[row] = base_lon + offset[1]
            master.loc[row, "approx"] = True
    master["lat"], master["lon"] = lat, lon
    return master.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_store_stats(_sales, signature, family):
    frame = _sales
    if "store_nbr" not in frame:
        return pd.DataFrame(columns=["store_nbr", "sales", "avg_daily", "families", "promo_share"])
    if family != ALL_FAMILY and "family" in frame:
        frame = frame[frame["family"].astype(str) == family]
    key = frame["store_nbr"].astype(str).rename("store_nbr")
    grouped = frame.groupby(key)
    out = grouped["sales"].sum().rename("sales").to_frame()
    days = grouped["date"].nunique().replace(0, np.nan)
    out["avg_daily"] = out["sales"] / days
    out["families"] = grouped["family"].nunique() if "family" in frame else 1
    if "onpromotion" in frame:
        flag = pd.to_numeric(frame["onpromotion"], errors="coerce").fillna(0) > 0
        out["promo_share"] = flag.groupby(key).mean() * 100
    else:
        out["promo_share"] = np.nan
    return out.reset_index()


@st.cache_data(show_spinner=False)
def compute_family_stats(_sales, signature, store):
    frame = _sales
    if store != ALL_STORE and "store_nbr" in frame:
        frame = frame[frame["store_nbr"].astype(str) == store]
    return abc_classification(frame) if len(frame) else pd.DataFrame()


def satellite_map(points: pd.DataFrame, value_col: str, value_label: str,
                  selected: str | None, show_labels: bool, show_places: bool) -> go.Figure:
    """Bản đồ vệ tinh Esri World Imagery, không cần token."""
    modern = hasattr(go, "Scattermap")  # plotly >= 5.24 (MapLibre)
    trace_cls = go.Scattermap if modern else go.Scattermapbox
    values = points[value_col].astype(float)
    spread = values.max() - values.min()
    scaled = (values - values.min()) / spread if spread > 0 else values * 0 + 0.5
    sizes = 10 + 26 * scaled

    hover = [
        f"<b>Cửa hàng {r.store_nbr}</b><br>{r.city}, {r.state}"
        f"<br>Loại {r.type} · Cụm {r.cluster}"
        f"<br>Tổng bán: {r.sales:,.0f}<br>TB/ngày: {r.avg_daily:,.0f}"
        f"<br>Số nhóm hàng: {int(r.families)}"
        + (f"<br>Có khuyến mãi: {r.promo_share:.1f}%" if pd.notna(r.promo_share) else "")
        for r in points.itertuples()
    ]
    fig = go.Figure()
    if selected and (points["store_nbr"] == selected).any():
        sel = points[points["store_nbr"] == selected]
        fig.add_trace(trace_cls(
            lat=sel["lat"], lon=sel["lon"], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=float(sizes[sel.index[0]]) + 22, color="#facc15", opacity=0.55),
        ))
    fig.add_trace(trace_cls(
        lat=points["lat"], lon=points["lon"],
        mode="markers+text" if show_labels else "markers",
        text=points["store_nbr"] if show_labels else None,
        textposition="top center",
        textfont=dict(color="white", size=11),
        hovertext=hover, hoverinfo="text", showlegend=False,
        marker=dict(
            size=sizes, color=values, colorscale="Turbo", opacity=0.92, showscale=True,
            colorbar=dict(title=dict(text=value_label), thickness=12, len=0.6, x=0.99),
        ),
    ))

    layers = [dict(below="traces", sourcetype="raster", source=[ESRI_IMAGERY],
                   sourceattribution="Imagery © Esri, Maxar, Earthstar Geographics")]
    if show_places:
        layers.append(dict(below="traces", sourcetype="raster", source=[ESRI_LABELS]))

    span = max(points["lat"].max() - points["lat"].min(),
               points["lon"].max() - points["lon"].min(), 0.05)
    zoom = float(max(3.0, min(12.0, math.log2(360 / span) - 0.6)))
    map_cfg = dict(
        style="white-bg",
        center=dict(lat=float(points["lat"].mean()), lon=float(points["lon"].mean())),
        zoom=zoom, layers=layers,
    )
    fig.update_layout(
        **{("map" if modern else "mapbox"): map_cfg},
        height=620, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ------------------------------------------------------------------ dữ liệu
page_intro(
    "Tổng quan doanh số, cửa hàng và tồn kho",
    "Theo dõi xu hướng bán hàng, vị trí cửa hàng trên bản đồ vệ tinh, phân loại ABC "
    "và trạng thái dự báo - tồn kho.",
)

sales = get_active_sales()
if show_data_required(sales):
    st.info("Mở trang **Nạp Dữ liệu** ở thanh bên để tải bộ dữ liệu hoặc tạo dữ liệu mẫu.")
    st.stop()

filtered = filter_controls(sales, key_prefix="dash")
if filtered.empty:
    st.warning("Không có dữ liệu cho bộ lọc đã chọn.")
    st.stop()

sig = _signature(sales)
tables = st.session_state.get("source_tables") or {}
master = build_store_master(sales, tables)
forecast = st.session_state.get("forecast_result")
metrics = st.session_state.get("test_metrics")
inventory = st.session_state.get("inventory_plan")
selected_family = st.session_state.get("selected_family", ALL_FAMILY)
selected_store = st.session_state.get("selected_store", ALL_STORE)

daily = aggregate_daily(filtered)
total_sales = float(filtered["sales"].fillna(0).sum())
stores_count = int(filtered["store_nbr"].nunique()) if "store_nbr" in filtered else 0
families_count = int(filtered["family"].nunique()) if "family" in filtered else 0
avg_daily = float(daily["sales"].mean()) if not daily.empty else 0.0

last28 = float(daily["sales"].tail(28).sum())
prev28 = float(daily["sales"].iloc[-56:-28].sum()) if len(daily) >= 56 else 0.0
growth_note = (
    f"28 ngày gần nhất {(last28 / prev28 - 1) * 100:+.1f}% so với kỳ trước"
    if prev28 > 0 else f"{daily['date'].min():%d/%m/%Y} – {daily['date'].max():%d/%m/%Y}"
)
top_family = (
    filtered.groupby("family")["sales"].sum().idxmax() if families_count else "—"
)
city_count = master["city"].nunique() if not master.empty else 0
kpi_cards([
    ("Tổng lượng bán", _fmt(total_sales), growth_note),
    ("Cửa hàng", f"{stores_count:,}",
     f"{city_count} thành phố · {master['state'].nunique()} tỉnh" if not master.empty else ""),
    ("Nhóm hàng", f"{families_count:,}", f"Bán chạy nhất: {top_family}"),
    ("Trung bình mỗi ngày", _fmt(avg_daily),
     f"Dự báo sắp tới: {_fmt(forecast['forecast'].sum())}"
     if forecast is not None and not forecast.empty else "Chưa có dự báo"),
])

tab_trend, tab_map, tab_rank, tab_inv = st.tabs(
    ["📈 Xu hướng", "🛰️ Bản đồ vệ tinh", "🏬 Cửa hàng & nhóm hàng", "📦 Dự báo & tồn kho"]
)

# ------------------------------------------------------------------ tab xu hướng
with tab_trend:
    freq_label = st.radio("Chu kỳ tổng hợp", ["Ngày", "Tuần", "Tháng"], horizontal=True)
    freq = {"Ngày": "D", "Tuần": "W", "Tháng": "MS"}[freq_label]
    series = daily.set_index("date")["sales"].resample(freq).sum().reset_index()

    left, right = st.columns([1.7, 1])
    with left:
        fig = go.Figure()
        fig.add_scatter(x=series["date"], y=series["sales"], mode="lines",
                        name="Thực tế", line=dict(color=CHART_COLORS[0], width=1.6))
        if freq == "D" and len(series) >= 7:
            fig.add_scatter(x=series["date"], y=series["sales"].rolling(7).mean(),
                            mode="lines", name="TB trượt 7 ngày",
                            line=dict(color=CHART_COLORS[2], width=2.4))
        if forecast is not None and not forecast.empty:
            fc = forecast.set_index("date")["forecast"].resample(freq).sum().reset_index()
            fig.add_scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Dự báo",
                            line=dict(dash="dash", color="#f97316", width=2.4))
        fig.update_layout(title=f"Doanh số theo {freq_label.lower()}", height=380,
                          xaxis_title=None, yaxis_title="Số lượng bán")
        st.plotly_chart(fig, width="stretch")
    with right:
        dow = daily.assign(dow=daily["date"].dt.dayofweek).groupby("dow")["sales"].mean()
        dow = dow.reindex(range(7)).fillna(0)
        fig_dow = px.bar(x=DOW_LABELS, y=dow.values, labels={"x": "Thứ", "y": "TB/ngày"},
                         title="Doanh số trung bình theo thứ")
        fig_dow.update_traces(marker_color=[CHART_COLORS[4] if v == dow.max() else CHART_COLORS[1]
                                            for v in dow.values])
        fig_dow.update_layout(height=380)
        st.plotly_chart(fig_dow, width="stretch")

    month = daily.assign(
        year=daily["date"].dt.year.astype(str), month=daily["date"].dt.month
    ).groupby(["year", "month"])["sales"].sum().reset_index()
    if month["year"].nunique() > 1 or month["month"].nunique() > 1:
        fig_m = px.line(month, x="month", y="sales", color="year", markers=True,
                        title="So sánh doanh số theo tháng giữa các năm",
                        labels={"month": "Tháng", "sales": "Số lượng bán", "year": "Năm"})
        fig_m.update_xaxes(dtick=1)
        fig_m.update_layout(height=340)
        st.plotly_chart(fig_m, width="stretch")

# ------------------------------------------------------------------ tab bản đồ
with tab_map:
    stats = compute_store_stats(sales, sig, selected_family)
    points = stats.merge(master, on="store_nbr", how="left") if not stats.empty else stats
    located = points.dropna(subset=["lat", "lon"]) if not points.empty else points

    if points.empty:
        st.info("Dữ liệu bán hàng chưa có cột `store_nbr` nên không thể vẽ bản đồ cửa hàng.")
    elif located.empty:
        st.warning(
            "Chưa xác định được vị trí cửa hàng. Hãy nạp bảng **stores** có cột `city` "
            "(hoặc `latitude`/`longitude`) ở trang Nạp Dữ liệu."
        )
    else:
        metric_options = {
            "Tổng doanh số": ("sales", "Tổng bán"),
            "Doanh số trung bình/ngày": ("avg_daily", "TB/ngày"),
            "Số nhóm hàng": ("families", "Nhóm hàng"),
        }
        if located["promo_share"].notna().any():
            metric_options["Tỷ lệ bán có khuyến mãi (%)"] = ("promo_share", "% KM")
        c1, c2, c3 = st.columns([2, 1, 1])
        metric_name = c1.selectbox("Chỉ số hiển thị (màu & kích thước)", list(metric_options))
        show_labels = c2.checkbox("Hiện số cửa hàng", value=True)
        show_places = c3.checkbox("Hiện tên địa danh", value=True)
        value_col, value_label = metric_options[metric_name]
        located = located.reset_index(drop=True)
        highlight = selected_store if selected_store != ALL_STORE else None

        map_col, side_col = st.columns([2.6, 1])
        with map_col:
            st.plotly_chart(
                satellite_map(located, value_col, value_label, highlight, show_labels, show_places),
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )
        with side_col:
            st.markdown(f"**Top 10 cửa hàng theo {metric_name.lower()}**")
            top = located.nlargest(10, value_col)[["store_nbr", "city", value_col]].rename(
                columns={"store_nbr": "Cửa hàng", "city": "Thành phố", value_col: value_label}
            )
            st.dataframe(top, hide_index=True, width="stretch")
            if located["approx"].any():
                st.caption(
                    "Vị trí được định vị theo trung tâm thành phố; các cửa hàng cùng thành phố "
                    "được rải quanh trung tâm để không chồng lên nhau. Thêm cột `latitude`/"
                    "`longitude` vào bảng stores để hiển thị đúng địa chỉ thực."
                )

        unlocated = points[points["lat"].isna() | points["lon"].isna()]
        if not unlocated.empty:
            names = ", ".join(sorted(unlocated["city"].fillna("Không rõ").astype(str).unique()))
            st.warning(f"{len(unlocated)} cửa hàng chưa định vị được (thành phố: {names}).")

        st.divider()
        st.subheader("Chi tiết cửa hàng")
        options = sorted(located["store_nbr"].unique(), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
        default = options.index(selected_store) if selected_store in options else 0
        store_id = st.selectbox("Chọn cửa hàng", options, index=default, key="dash_detail_store")
        info = located[located["store_nbr"] == store_id].iloc[0]
        rank = int(located["sales"].rank(ascending=False, method="min")[located["store_nbr"] == store_id].iloc[0])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Vị trí", f"{info['city']}", f"{info['state']}", delta_color="off")
        d2.metric("Loại · Cụm", f"{info['type']} · {info['cluster']}")
        d3.metric("Tổng bán", _fmt(info["sales"]),
                  f"Hạng {rank}/{len(located)} · {info['sales'] / located['sales'].sum() * 100:.1f}% tổng",
                  delta_color="off")
        d4.metric("TB/ngày", _fmt(info["avg_daily"]))
        st.caption(f"Tọa độ: {info['lat']:.4f}, {info['lon']:.4f}")

        store_frame = sales[sales["store_nbr"].astype(str) == store_id]
        if selected_family != ALL_FAMILY and "family" in store_frame:
            store_frame = store_frame[store_frame["family"].astype(str) == selected_family]
        g1, g2 = st.columns([1.6, 1])
        with g1:
            sd = aggregate_daily(store_frame)
            sd = sd.set_index("date")["sales"].resample("W").sum().reset_index()
            fig_s = px.area(sd, x="date", y="sales", title=f"Doanh số tuần – cửa hàng {store_id}",
                            labels={"date": "Tuần", "sales": "Số lượng bán"})
            fig_s.update_traces(line_color=CHART_COLORS[1])
            fig_s.update_layout(height=320)
            st.plotly_chart(fig_s, width="stretch")
        with g2:
            if "family" in store_frame:
                tf = (store_frame.groupby("family")["sales"].sum().nlargest(6)
                      .rename("sales").reset_index())
                fig_f = px.bar(tf, x="sales", y="family", orientation="h",
                               title="Nhóm hàng bán chạy",
                               labels={"sales": "Số lượng bán", "family": ""})
                fig_f.update_layout(height=320, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_f, width="stretch")

# ------------------------------------------------------------------ tab cửa hàng & nhóm hàng
with tab_rank:
    stats_all = compute_store_stats(sales, sig, selected_family)
    merged = stats_all.merge(master, on="store_nbr", how="left") if not stats_all.empty else stats_all

    a, b = st.columns(2)
    with a:
        if not merged.empty:
            top_s = merged.nlargest(10, "sales")
            fig = px.bar(top_s, x="store_nbr", y="sales", color="city",
                         title="Top 10 cửa hàng theo doanh số",
                         labels={"store_nbr": "Cửa hàng", "sales": "Số lượng bán", "city": "Thành phố"})
            fig.update_xaxes(type="category")
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch")
    with b:
        if not merged.empty and "city" in merged:
            by_city = merged.groupby("city")["sales"].sum().nlargest(10).reset_index()
            fig = px.bar(by_city, x="sales", y="city", orientation="h",
                         title="Doanh số theo thành phố (top 10)",
                         labels={"sales": "Số lượng bán", "city": ""})
            fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2)
    with c:
        if not merged.empty and merged["type"].nunique() > 1:
            by_type = merged.groupby("type")["sales"].sum().reset_index()
            fig = px.pie(by_type, names="type", values="sales", hole=0.5,
                         title="Cơ cấu doanh số theo loại cửa hàng")
            fig.update_layout(height=360)
            st.plotly_chart(fig, width="stretch")
    with d:
        fam = compute_family_stats(sales, sig, selected_store)
        if not fam.empty:
            fig = px.bar(fam.head(12), x="sales", y="family", orientation="h", color="abc_class",
                         title="Nhóm hàng & phân loại ABC (top 12)",
                         color_discrete_map={"A": "#10b981", "B": "#f59e0b", "C": "#fb7185"},
                         labels={"sales": "Số lượng bán", "family": "", "abc_class": "Hạng"})
            fig.update_layout(height=360, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

    fam = compute_family_stats(sales, sig, selected_store)
    if not fam.empty:
        counts = fam["abc_class"].value_counts()
        st.caption(
            "Phân loại ABC theo đóng góp lũy kế: "
            + " · ".join(f"hạng {k}: {int(counts.get(k, 0))} nhóm" for k in "ABC")
        )
        with st.expander("Bảng phân loại ABC đầy đủ"):
            st.dataframe(
                fam.rename(columns={"family": "Nhóm hàng", "sales": "Tổng bán",
                                    "cum_pct": "Lũy kế (%)", "abc_class": "Hạng"}),
                hide_index=True, width="stretch",
            )

# ------------------------------------------------------------------ tab dự báo & tồn kho
with tab_inv:
    m_col, i_col = st.columns(2)
    with m_col:
        st.subheader("Trạng thái mô hình")
        if metrics:
            cfg = st.session_state.get("model_config") or {}
            if cfg:
                st.caption(f"{cfg.get('type', '')} · look-back {cfg.get('look_back', '')} ngày · "
                           f"{cfg.get('units', '')} units · {cfg.get('layers', '')} lớp")
            k1, k2 = st.columns(2)
            k1.metric("RMSE kiểm định", f"{metrics['RMSE']:,.2f}")
            k2.metric("MAE kiểm định", f"{metrics['MAE']:,.2f}")
            k3, k4 = st.columns(2)
            k3.metric("MAPE kiểm định", f"{metrics['MAPE (%)']:,.2f}%")
            if "WAPE (%)" in metrics:
                k4.metric("WAPE kiểm định", f"{metrics['WAPE (%)']:,.2f}%")
        else:
            st.info("Chưa có kết quả đánh giá. Huấn luyện mô hình để bắt đầu.")

        if forecast is not None and not forecast.empty:
            st.markdown("**Tóm tắt dự báo**")
            peak = forecast.loc[forecast["forecast"].idxmax()]
            st.write(
                f"- Giai đoạn: {forecast['date'].min():%d/%m/%Y} – {forecast['date'].max():%d/%m/%Y} "
                f"({len(forecast)} ngày)\n"
                f"- Tổng dự báo: **{forecast['forecast'].sum():,.0f}** · "
                f"TB/ngày: **{forecast['forecast'].mean():,.0f}**\n"
                f"- Đỉnh: **{peak['forecast']:,.0f}** vào {peak['date']:%d/%m/%Y}"
            )
    with i_col:
        st.subheader("Tồn kho")
        if inventory:
            tag = status_tag("Cần đặt hàng", "bad") if inventory["need_reorder"] \
                else status_tag("Chưa cần đặt hàng", "ok")
            st.markdown(f"Khuyến nghị: {tag}", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            r1.metric("Điểm đặt hàng lại (ROP)", f"{inventory['reorder_point']:,.1f}")
            r2.metric("Tồn kho an toàn", f"{inventory['safety_stock']:,.1f}")
            r3, r4 = st.columns(2)
            days = inventory["days_of_supply"]
            r3.metric("Số ngày tồn kho đủ bán", "∞" if not np.isfinite(days) else f"{days:,.1f}")
            r4.metric("EOQ", f"{inventory['eoq']:,.1f}" if inventory.get("eoq") else "—")
            if inventory["need_reorder"]:
                st.warning(f"Số lượng đặt đề xuất: **{inventory['suggested_order_qty']:,.0f}**")
            st.caption(f"Nhu cầu trong lead time: {inventory['lead_time_demand']:,.0f} · "
                       f"Tồn cuối kỳ dự kiến: {inventory['projected_ending_stock']:,.0f}")
        else:
            st.info("Chưa có tham số tồn kho. Vào trang **Quản trị tồn kho** để tính ROP, EOQ.")