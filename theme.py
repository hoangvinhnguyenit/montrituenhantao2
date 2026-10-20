# -*- coding: utf-8 -*-
"""Giao diện Streamlit đồng bộ với index.html / Style.css (menu màu sáng, thẻ KPI, thanh tiêu đề xanh)."""

from __future__ import annotations

import html

import streamlit as st

TITLE = "DỰ BÁO DOANH SỐ & QUẢN TRỊ TỒN KHO"
SUBTITLE = "NHÓM 1: Nguyễn Thị Diệu Hiền - Nguyễn Hoàng Vinh"

# Màu cho 8 trang theo thứ tự trong st.navigation (khớp bảng màu c1..c7 của index.html)
MENU_COLORS = [
    ("#34d399", "#059669"),  # Dashboard
    ("#fb923c", "#ea580c"),  # Dữ liệu
    ("#fbbf24", "#d97706"),  # Tiền xử lý
    ("#818cf8", "#4338ca"),  # Phân tích EDA
    ("#a78bfa", "#7c3aed"),  # Huấn luyện mô hình
    ("#38bdf8", "#0284c7"),  # Dự báo doanh số
    ("#fb7185", "#e11d48"),  # Quản trị tồn kho
    ("#2dd4bf", "#0f766e"),  # Báo cáo tổng hợp
]
KPI_GRADIENTS = [
    ("#10b981", "#047857"),
    ("#38bdf8", "#0369a1"),
    ("#a78bfa", "#6d28d9"),
    ("#fb7185", "#be123c"),
]
CHART_COLORS = ["#10b981", "#38bdf8", "#f59e0b", "#a78bfa", "#fb7185", "#14b8a6", "#f97316", "#6366f1"]

_NAV = '[data-testid="stSidebarNav"]'
_COL = ':is([data-testid="stColumn"],[data-testid="column"])'


def _build_css(menu_colors=None) -> str:
    palette = MENU_COLORS if menu_colors is None else menu_colors
    menu_vars = "\n".join(
        f"{_NAV} li:nth-child({i}){{--c1:{a};--c2:{b}}}"
        for i, (a, b) in enumerate(palette, start=1)
    )
    kpi = "\n".join(
        f"{_COL}:nth-child({i}) [data-testid=\"stMetric\"]{{background:linear-gradient(135deg,{a},{b})}}"
        for i, (a, b) in enumerate(KPI_GRADIENTS, start=1)
    )
    return (
        """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp,.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,.stApp li,.stApp td,.stApp th,
.stApp input,.stApp textarea,.stApp button{font-family:'Plus Jakarta Sans',system-ui,sans-serif!important}
.stApp,[data-testid="stAppViewContainer"]{background:#f1f5f9}
.block-container{padding-top:7.2rem!important;max-width:1680px}
footer,[data-testid="stAppDeployButton"],#MainMenu{display:none!important}
div[data-testid="stElementContainer"]:has(> div > style),div[data-testid="stElementContainer"]:has(> .stMarkdown style){display:none}

/* ===== Thanh tiêu đề xanh (giống .topbar) - phủ toàn chiều ngang, nằm trên sidebar ===== */
html::before{content:"";position:fixed;top:0;left:0;width:100vw;height:92px;
  box-sizing:border-box;z-index:1000000;background:linear-gradient(90deg,#094a1d,#0d5c24 50%,#08431a);
  border-bottom:3px solid #ca8a04;box-shadow:0 6px 24px -4px rgba(9,74,29,.45);pointer-events:none}
html::after{content:"__TITLE__";position:fixed;left:1.75rem;top:14px;z-index:1000001;color:#fff;font-weight:800;
  font-family:'Plus Jakarta Sans',system-ui,sans-serif;
  font-size:clamp(16px,2.4vw,30px);line-height:1.2;letter-spacing:.04em;white-space:nowrap;
  text-shadow:0 2px 6px rgba(0,0,0,.3);pointer-events:none}
body::before{content:"__SUBTITLE__";position:fixed;left:1.75rem;top:54px;z-index:1000001;color:#facc15;font-weight:600;
  font-family:'Plus Jakarta Sans',system-ui,sans-serif;
  font-size:clamp(11px,1.3vw,15px);letter-spacing:.06em;pointer-events:none}
[data-testid="stHeader"]{height:92px;background:transparent!important;border:0!important;box-shadow:none!important;z-index:1000002}
[data-testid="stHeader"]::before,[data-testid="stHeader"]::after{content:none!important}
[data-testid="stHeader"] svg,[data-testid="stToolbar"] svg{color:#fff!important;fill:#fff!important}
[data-testid="stHeader"] button{color:#fff!important}

/* ===== Sidebar và menu nút màu sáng (giống .mbtn) ===== */
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2e8f0;box-shadow:0 14px 30px -14px rgba(15,23,42,.18)}
[data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="stSidebarContent"]{padding:6.2rem .8rem 1rem}
[data-testid="stSidebarNavItems"]{gap:0!important;padding:0!important;max-height:none!important}
__NAV__ li{--c1:#22c55e;--c2:#16a34a;list-style:none;margin:0 0 9px}
__MENUVARS__
[data-testid="stSidebarNavLink"]{background:linear-gradient(135deg,var(--c1,#22c55e),var(--c2,#16a34a))!important;
  color:#fff!important;border-radius:16px!important;padding:10px 16px!important;border:2px solid rgba(255,255,255,.55)!important;
  box-shadow:0 6px 16px -4px var(--c2,#16a34a);transition:transform .18s,box-shadow .18s,filter .18s}
[data-testid="stSidebarNavLink"] *{color:#fff!important;font-weight:800!important;font-size:15px}
[data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"]{font-size:24px!important}
[data-testid="stSidebarNavLink"]:hover{transform:translateY(-3px) scale(1.02);filter:brightness(1.08);
  box-shadow:0 12px 24px -6px var(--c2,#16a34a)}
[data-testid="stSidebarNavLink"][aria-current="page"]{outline:4px solid #facc15;outline-offset:2px;
  box-shadow:0 0 0 6px rgba(250,204,21,.25),0 12px 26px -6px var(--c2,#16a34a)}

/* ===== Thẻ chỉ số (st.metric) dạng KPI màu ===== */
[data-testid="stMetric"]{position:relative;overflow:hidden;border-radius:22px;padding:18px 20px;color:#fff;
  background:linear-gradient(135deg,#10b981,#047857);box-shadow:0 12px 24px -10px rgba(0,0,0,.4)}
__KPI__
[data-testid="stMetric"] *,[data-testid="stMetricLabel"] *{color:#fff!important}
[data-testid="stMetricLabel"]{font-weight:600;opacity:.95}
[data-testid="stMetricValue"]{font-size:30px!important;font-weight:800!important;line-height:1.1}
[data-testid="stMetricDelta"]{opacity:.92}

/* ===== Thẻ nội dung (st.container(border=True), expander) ===== */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:24px}
[data-testid="stExpander"],div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]{
  background:rgba(255,255,255,.9);border:1px solid #e2e8f0!important;border-radius:24px!important;box-shadow:0 14px 30px -14px rgba(15,23,42,.18)}
h1,h2,h3{font-weight:800!important;color:#0f172a}
[data-testid="stDataFrame"]{border:1px solid #e2e8f0;border-radius:16px;overflow:hidden}

/* ===== Thông báo ===== */
[data-testid="stAlert"]{border-radius:16px;border:1px solid rgba(0,0,0,.06);font-size:14px}

/* ===== Nút bấm (giống .btn) ===== */
button[data-testid^="stBaseButton"]{background:#0d5c24!important;border:2px solid #0d5c24!important;border-radius:10px!important;
  font-weight:700!important;padding:.45rem 1rem!important}
button[data-testid^="stBaseButton"] *{color:#fff!important}
button[data-testid^="stBaseButton"]:hover{background:#094a1d!important;transform:translateY(-1px)}
[data-testid="stHeader"] button[data-testid^="stBaseButton"]{background:transparent!important;border:none!important}

/* ===== Ô chọn, nhập liệu ===== */
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]{border-radius:10px!important;border:2px solid #cbd5e1!important;background:#fff}
[data-testid="stTabs"] button[aria-selected="true"]{color:#0d5c24!important;font-weight:800}
[data-baseweb="tab-highlight"]{background:#facc15!important;height:4px!important}

/* ===== Khung tải file (giống .drop) ===== */
[data-testid="stFileUploaderDropzone"]{border:2px dashed #a5b4fc;border-radius:24px;padding:36px 16px;
  background:linear-gradient(180deg,rgba(238,242,255,.5),rgba(224,231,255,.35));transition:.2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#4f46e5;background:rgba(224,231,255,.8)}

/* ===== Thẻ HTML từ helper (kpi_cards, page_intro) ===== */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:1rem}
.kpi{position:relative;overflow:hidden;border-radius:22px;padding:18px 20px;color:#fff;box-shadow:0 12px 24px -10px rgba(0,0,0,.4)}
.kpi span{font-size:12px;font-weight:600;opacity:.95}
.kpi b{display:block;font-size:30px;font-weight:800;line-height:1.1;margin-top:6px}
.kpi em{display:block;font-style:normal;font-size:12px;margin-top:6px;opacity:.9}
.page-intro{background:rgba(255,255,255,.9);border:1px solid #e2e8f0;border-left:5px solid #16a34a;border-radius:24px;
  padding:18px 22px;margin-bottom:1rem;box-shadow:0 14px 30px -14px rgba(15,23,42,.18)}
.page-intro h2{margin:0;font-size:clamp(22px,3vw,30px);font-weight:800;color:#0f172a}
.page-intro p{margin:6px 0 0;color:#64748b;font-size:14px}
.tag{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#fff}
.t-ok{background:#16a34a}.t-warn{background:#d97706}.t-bad{background:#dc2626}
"""
        .replace("__TITLE__", TITLE)
        .replace("__SUBTITLE__", SUBTITLE)
        .replace("__NAV__", _NAV)
        .replace("__MENUVARS__", menu_vars)
        .replace("__KPI__", kpi)
    )


def _register_plotly_template() -> None:
    """Đặt bảng màu và font cho Plotly. Gọi st.plotly_chart(fig, theme=None) để dùng template này."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return
    grid = "rgba(15,23,42,.08)"
    pio.templates["favorita"] = go.layout.Template(
        layout=dict(
            colorway=CHART_COLORS,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#475569"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=grid, zeroline=False),
            yaxis=dict(gridcolor=grid, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=20, t=50, b=40),
        )
    )
    pio.templates.default = "favorita"


def apply_theme(menu_colors=None) -> None:
    """Gọi một lần trong app.py, trước st.navigation(...).run()."""
    st.markdown(f"<style>{_build_css(menu_colors)}</style>", unsafe_allow_html=True)
    _register_plotly_template()


# ---------- Hàm trợ giúp cho các trang (tùy chọn) ----------
def page_intro(title: str, description: str = "") -> None:
    """Khung tiêu đề trang có viền xanh bên trái (giống .page-intro)."""
    st.markdown(
        f'<div class="page-intro"><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_cards(items) -> None:
    """items: danh sách (nhãn, giá trị, ghi chú). Tối đa 4 thẻ có màu riêng, từ thẻ 5 lặp lại màu."""
    cards = []
    for i, item in enumerate(items):
        label, value, *rest = item
        note = rest[0] if rest else ""
        a, b = KPI_GRADIENTS[i % len(KPI_GRADIENTS)]
        cards.append(
            f'<div class="kpi" style="background:linear-gradient(135deg,{a},{b})">'
            f"<span>{html.escape(str(label))}</span><b>{html.escape(str(value))}</b>"
            f"<em>{html.escape(str(note))}</em></div>"
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def status_tag(label: str, level: str = "ok") -> str:
    """Trả về HTML nhãn trạng thái. level: ok | warn | bad."""
    return f'<span class="tag t-{level}">{html.escape(label)}</span>'