# -*- coding: utf-8 -*-
"""Trang dành cho admin: tạo user và tích chọn quyền cho 8 chức năng."""

import pandas as pd
import streamlit as st

import auth
from app_common import configure_page, initialize_session_state

configure_page("Quản lý người dùng")
initialize_session_state()

me = auth.current_user()
if me is None or me["role"] != auth.ROLE_ADMIN:
    st.error("Chỉ quản trị viên mới được truy cập trang này.")
    st.stop()

st.title("Quản lý người dùng & phân quyền")
st.caption(
    "Tích chọn từng chức năng để cho phép user sử dụng. Admin luôn có đủ quyền. "
    "Thay đổi có hiệu lực ngay ở lần thao tác tiếp theo của user."
)

# ------------------------------------------------------------ bảng phân quyền
st.subheader("Bảng phân quyền")
users = auth.list_users()
rows = []
for u in users:
    is_admin = u["role"] == auth.ROLE_ADMIN
    row = {
        "Tên đăng nhập": u["username"],
        "Họ tên": u["display_name"],
        "Vai trò": u["role"],
        "Hoạt động": u["active"],
    }
    for key in auth.PAGE_KEYS:
        row[auth.PAGE_TITLES[key]] = True if is_admin else key in u["permissions"]
    rows.append(row)

table = pd.DataFrame(rows)
page_columns = [auth.PAGE_TITLES[k] for k in auth.PAGE_KEYS]
column_config = {
    "Vai trò": st.column_config.SelectboxColumn("Vai trò", options=list(auth.ROLES), required=True),
    "Hoạt động": st.column_config.CheckboxColumn("Hoạt động"),
}
for title in page_columns:
    column_config[title] = st.column_config.CheckboxColumn(title)

edited = st.data_editor(
    table,
    column_config=column_config,
    disabled=["Tên đăng nhập"],
    hide_index=True,
    width="stretch",
    key="permission_editor",
)

if st.button("Lưu phân quyền", type="primary"):
    errors = []
    for _, row in edited.iterrows():
        username = row["Tên đăng nhập"]
        role = row["Vai trò"]
        active = bool(row["Hoạt động"])
        if username == me["username"] and (role != auth.ROLE_ADMIN or not active):
            errors.append(f"Không thể tự hạ quyền hoặc khóa chính mình ({username}).")
            continue
        permissions = [
            key for key in auth.PAGE_KEYS if bool(row[auth.PAGE_TITLES[key]])
        ]
        original = next(u for u in users if u["username"] == username)
        if original["role"] == auth.ROLE_ADMIN and role == auth.ROLE_ADMIN:
            permissions = None  # admin luôn đủ quyền, giữ nguyên bản ghi
        try:
            auth.update_user(
                username,
                display_name=row["Họ tên"],
                role=role,
                active=active,
                permissions=permissions,
            )
        except ValueError as exc:
            errors.append(f"{username}: {exc}")
    if errors:
        for message in errors:
            st.error(message)
    else:
        st.success("Đã lưu phân quyền.")
        st.rerun()

# ------------------------------------------------------------ tạo user
st.divider()
st.subheader("Tạo người dùng mới")
with st.form("create_user_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    new_username = c1.text_input("Tên đăng nhập")
    new_display = c2.text_input("Họ tên")
    c3, c4 = st.columns(2)
    new_password = c3.text_input("Mật khẩu (tối thiểu 6 ký tự)", type="password")
    new_role = c4.selectbox("Vai trò", list(auth.ROLES))
    new_permissions = st.multiselect(
        "Chức năng được phép sử dụng (áp dụng cho vai trò user)",
        options=auth.PAGE_KEYS,
        format_func=lambda key: auth.PAGE_TITLES[key],
    )
    if st.form_submit_button("Tạo người dùng"):
        try:
            auth.create_user(
                new_username, new_password, new_display, new_role,
                auth.PAGE_KEYS if new_role == auth.ROLE_ADMIN else new_permissions,
            )
            st.success(f"Đã tạo tài khoản {new_username.strip().lower()}.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

# ------------------------------------------------------------ đặt lại mật khẩu / xóa
st.divider()
st.subheader("Đặt lại mật khẩu hoặc xóa tài khoản")
usernames = [u["username"] for u in users]
target = st.selectbox("Chọn tài khoản", usernames)
col_reset, col_delete = st.columns(2)
with col_reset:
    with st.form("reset_password_form", clear_on_submit=True):
        temp_password = st.text_input("Mật khẩu mới", type="password")
        if st.form_submit_button("Đặt lại mật khẩu"):
            try:
                auth.set_password(target, temp_password, must_change=True)
                st.success("Đã đặt lại. User sẽ được nhắc đổi mật khẩu khi đăng nhập.")
            except ValueError as exc:
                st.error(str(exc))
with col_delete:
    st.write("")
    confirm_delete = st.checkbox(f"Xác nhận xóa tài khoản **{target}**")
    if st.button("Xóa tài khoản", disabled=not confirm_delete):
        if target == me["username"]:
            st.error("Không thể xóa chính tài khoản đang đăng nhập.")
        else:
            try:
                auth.delete_user(target)
                st.success(f"Đã xóa {target}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
