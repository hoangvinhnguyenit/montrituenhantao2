# -*- coding: utf-8 -*-
"""Streamlit page router: đăng nhập + phân quyền theo từng chức năng."""

import streamlit as st

import auth
from theme import MENU_COLORS, apply_theme

st.set_page_config(
    page_title="Dự báo doanh số & quản trị tồn kho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    auth.init_store()
    user = auth.current_user()
except RuntimeError as exc:
    apply_theme(menu_colors=[])
    st.error(f"Không thể kết nối kho tài khoản: {exc}")
    st.stop()

if user is None:
    apply_theme(menu_colors=[])
    auth.render_login()
    st.stop()

# Chỉ đăng ký những trang user được phép; trang không đăng ký thì không truy cập được.
allowed = set(auth.allowed_page_keys(user))
pages, colors = [], []
for index, page in enumerate(auth.PAGES):
    if page["key"] in allowed:
        pages.append(
            st.Page(
                page["path"],
                title=page["title"],
                icon=page["icon"],
                default=not pages,  # trang đầu tiên được phép làm trang mặc định
            )
        )
        colors.append(MENU_COLORS[index])  # giữ đúng màu của từng chức năng

if user["role"] == auth.ROLE_ADMIN:
    pages.append(
        st.Page(
            "app_pages/8_Quan_tri_nguoi_dung.py",
            title="Quản lý người dùng",
            icon=":material/admin_panel_settings:",
            default=not pages,
        )
    )
    colors.append(("#94a3b8", "#475569"))

apply_theme(menu_colors=colors)

if not pages:
    st.warning(
        "Tài khoản của bạn chưa được cấp quyền sử dụng chức năng nào. "
        "Vui lòng liên hệ quản trị viên."
    )
    auth.render_account_sidebar(user)
    st.stop()

auth.render_account_sidebar(user)
st.navigation(pages, position="sidebar").run()
