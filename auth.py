# -*- coding: utf-8 -*-
"""Xác thực và phân quyền: admin cấp quyền cho user theo 8 chức năng của menu."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

TABLE = "app_users"
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLES = (ROLE_USER, ROLE_ADMIN)
ITERATIONS = 240_000
MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 60

# 8 chức năng (khớp thứ tự và tên trong menu). key dùng để lưu quyền.
PAGES = [
    {"key": "dashboard", "path": "app_pages/0_Dashboard.py", "title": "Tổng quan", "icon": ":material/dashboard:"},
    {"key": "du_lieu", "path": "app_pages/1_Du_lieu.py", "title": "Nạp Dữ liệu", "icon": ":material/table_view:"},
    {"key": "tien_xu_ly", "path": "app_pages/2_Tien_xu_ly.py", "title": "Xử lý dữ liệu", "icon": ":material/tune:"},
    {"key": "eda", "path": "app_pages/3_Phan_tich_EDA.py", "title": "Phân tích dữ liệu (EDA)", "icon": ":material/analytics:"},
    {"key": "huan_luyen", "path": "app_pages/4_Huan_luyen.py", "title": "Thiết lập mô hình", "icon": ":material/model_training:"},
    {"key": "du_bao", "path": "app_pages/5_Du_bao.py", "title": "Dự báo doanh số", "icon": ":material/trending_up:"},
    {"key": "ton_kho", "path": "app_pages/6_Quan_tri_ton_kho.py", "title": "Quản trị tồn kho", "icon": ":material/inventory_2:"},
    {"key": "bao_cao", "path": "app_pages/7_Bao_cao.py", "title": "Báo cáo tổng hợp", "icon": ":material/description:"},
]
PAGE_KEYS = [page["key"] for page in PAGES]
PAGE_TITLES = {page["key"]: page["title"] for page in PAGES}



# ---------------------------------------------------------------- mật khẩu
def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    )
    return salt, digest.hex()


def _verify_password(password: str, record: dict) -> bool:
    _, digest = _hash_password(password, record["salt"])
    return hmac.compare_digest(digest, record["hash"])


# ---------------------------------------------------------------- lưu trữ (Supabase)
def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return None
    return str(value) if value else None


@st.cache_resource(show_spinner=False)
def _client():
    url = _secret("SUPABASE_URL")
    # Bảng tài khoản chứa hash mật khẩu nên cần service-role key (chỉ dùng phía máy chủ).
    key = _secret("SUPABASE_SERVICE_KEY") or _secret("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Chưa cấu hình SUPABASE_URL và SUPABASE_SERVICE_KEY cho bảng tài khoản."
        )
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("Thiếu gói supabase. Cài bằng `pip install supabase`.") from exc
    return create_client(url, key)


def _run(query):
    try:
        return query.execute().data or []
    except Exception as exc:  # APIError, lỗi mạng...
        text = str(exc)
        if "PGRST205" in text or "does not exist" in text:
            raise RuntimeError(
                f"Chưa có bảng {TABLE}. Hãy chạy file supabase_app_users.sql trong Supabase."
            ) from exc
        if "permission denied" in text.lower() or "42501" in text:
            raise RuntimeError(
                "Không đủ quyền đọc bảng tài khoản. Hãy đặt SUPABASE_SERVICE_KEY "
                "(service-role key) trong secrets của máy chủ."
            ) from exc
        raise RuntimeError(f"Lỗi truy cập bảng {TABLE}: {text}") from exc


def _to_record(row: dict) -> dict:
    return {
        "display_name": row.get("display_name") or "",
        "role": row["role"],
        "active": bool(row["active"]),
        "permissions": list(row.get("permissions") or []),
        "salt": row["password_salt"],
        "hash": row["password_hash"],
        "must_change": bool(row.get("must_change", False)),
    }


def _fetch_all() -> dict:
    rows = _run(_client().table(TABLE).select("*").order("username"))
    return {row["username"]: _to_record(row) for row in rows}


def _fetch_one(username: str) -> dict | None:
    rows = _run(_client().table(TABLE).select("*").eq("username", username).limit(1))
    return _to_record(rows[0]) if rows else None


def _new_row(username, password, display_name, role, permissions, must_change=False) -> dict:
    salt, digest = _hash_password(password)
    return {
        "username": username,
        "display_name": display_name,
        "role": role,
        "active": True,
        "permissions": [k for k in permissions if k in PAGE_KEYS],
        "password_salt": salt,
        "password_hash": digest,
        "must_change": must_change,
    }


def _check_admin_left(users: dict) -> None:
    if not any(u["role"] == ROLE_ADMIN and u["active"] for u in users.values()):
        raise ValueError("Phải còn ít nhất một admin đang hoạt động.")


@st.cache_resource(show_spinner=False)
def _bootstrap() -> bool:
    """Tạo admin đầu tiên nếu bảng còn trống. Chỉ chạy một lần mỗi tiến trình."""
    if not _run(_client().table(TABLE).select("username").limit(1)):
        env_password = _secret("ADMIN_PASSWORD")
        _run(
            _client().table(TABLE).insert(
                _new_row(
                    "admin", env_password or "admin123", "Quản trị viên",
                    ROLE_ADMIN, PAGE_KEYS, must_change=not env_password,
                )
            )
        )
    return True


def init_store() -> None:
    _bootstrap()


def _public(username: str, record: dict) -> dict:
    return {
        "username": username,
        "display_name": record.get("display_name") or username,
        "role": record["role"],
        "active": record["active"],
        "permissions": list(record.get("permissions", [])),
        "must_change": record.get("must_change", False),
    }


def list_users() -> list[dict]:
    return [_public(name, rec) for name, rec in _fetch_all().items()]


def get_user(username: str) -> dict | None:
    record = _fetch_one(username)
    return _public(username, record) if record else None


def authenticate(username: str, password: str) -> dict | None:
    username = username.strip().lower()
    record = _fetch_one(username) if username else None
    if record is None or not record["active"]:
        _hash_password(password)  # giữ thời gian phản hồi tương đương
        return None
    return _public(username, record) if _verify_password(password, record) else None


def create_user(username, password, display_name, role, permissions) -> None:
    username = username.strip().lower()
    if not username or not username.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Tên đăng nhập chỉ gồm chữ, số, dấu chấm hoặc gạch dưới.")
    if len(password) < 6:
        raise ValueError("Mật khẩu phải có ít nhất 6 ký tự.")
    if role not in ROLES:
        raise ValueError("Vai trò không hợp lệ.")
    if _fetch_one(username) is not None:
        raise ValueError("Tên đăng nhập đã tồn tại.")
    _run(
        _client().table(TABLE).insert(
            _new_row(username, password, display_name or username, role, permissions)
        )
    )


def update_user(username, *, display_name=None, role=None, active=None, permissions=None) -> None:
    users = _fetch_all()
    record = users.get(username)
    if record is None:
        raise ValueError("Không tìm thấy người dùng.")
    changes = {}
    if display_name is not None:
        changes["display_name"] = display_name
    if role is not None:
        if role not in ROLES:
            raise ValueError("Vai trò không hợp lệ.")
        changes["role"] = role
    if active is not None:
        changes["active"] = bool(active)
    if permissions is not None:
        changes["permissions"] = [k for k in permissions if k in PAGE_KEYS]
    if not changes:
        return
    users[username] = {**record, **changes}
    _check_admin_left(users)
    _run(_client().table(TABLE).update(changes).eq("username", username))


def set_password(username: str, new_password: str, must_change: bool = False) -> None:
    if len(new_password) < 6:
        raise ValueError("Mật khẩu phải có ít nhất 6 ký tự.")
    if _fetch_one(username) is None:
        raise ValueError("Không tìm thấy người dùng.")
    salt, digest = _hash_password(new_password)
    _run(
        _client().table(TABLE)
        .update({"password_salt": salt, "password_hash": digest, "must_change": must_change})
        .eq("username", username)
    )


def delete_user(username: str) -> None:
    users = _fetch_all()
    if username not in users:
        raise ValueError("Không tìm thấy người dùng.")
    del users[username]
    _check_admin_left(users)
    _run(_client().table(TABLE).delete().eq("username", username))


# ---------------------------------------------------------------- phiên đăng nhập
def current_user() -> dict | None:
    """Đọc lại từ kho mỗi lần chạy, nên đổi quyền/khóa tài khoản có hiệu lực ngay."""
    username = st.session_state.get("auth_user")
    if not username:
        return None
    user = get_user(username)
    if user is None or not user["active"]:
        st.session_state.clear()
        return None
    return user


def allowed_page_keys(user: dict) -> list[str]:
    if user["role"] == ROLE_ADMIN:
        return list(PAGE_KEYS)
    return [key for key in PAGE_KEYS if key in user["permissions"]]


def logout() -> None:
    st.session_state.clear()
    st.rerun()


# ---------------------------------------------------------------- giao diện
def render_login() -> None:
    st.markdown(
        "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']"
        "{display:none!important}</style>",
        unsafe_allow_html=True,
    )
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.subheader("Đăng nhập hệ thống")
        locked_until = st.session_state.get("login_locked_until", 0)
        remaining = int(locked_until - time.time())
        if remaining > 0:
            st.error(f"Nhập sai quá nhiều lần. Thử lại sau {remaining} giây.")
            return
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập", type="primary", width="stretch")
        if submitted:
            try:
                user = authenticate(username, password)
            except RuntimeError as exc:
                st.error(str(exc))
                return
            if user is None:
                fails = st.session_state.get("login_fails", 0) + 1
                st.session_state["login_fails"] = fails
                if fails >= MAX_FAILED_LOGINS:
                    st.session_state["login_locked_until"] = time.time() + LOCK_SECONDS
                    st.session_state["login_fails"] = 0
                st.error("Sai tên đăng nhập hoặc mật khẩu, hoặc tài khoản đã bị khóa.")
            else:
                st.session_state.clear()
                st.session_state["auth_user"] = user["username"]
                st.rerun()


def render_account_sidebar(user: dict) -> None:
    with st.sidebar:
        st.divider()
        role_label = "Quản trị viên" if user["role"] == ROLE_ADMIN else "Người dùng"
        st.markdown(f"**{user['display_name']}**  \n{role_label} · `{user['username']}`")
        if user["must_change"]:
            st.warning("Bạn đang dùng mật khẩu mặc định. Hãy đổi mật khẩu ngay.")
        with st.expander("Đổi mật khẩu", expanded=user["must_change"]):
            with st.form("change_password_form", clear_on_submit=True):
                old = st.text_input("Mật khẩu hiện tại", type="password")
                new = st.text_input("Mật khẩu mới", type="password")
                confirm = st.text_input("Nhập lại mật khẩu mới", type="password")
                if st.form_submit_button("Cập nhật"):
                    if authenticate(user["username"], old) is None:
                        st.error("Mật khẩu hiện tại không đúng.")
                    elif new != confirm:
                        st.error("Mật khẩu nhập lại không khớp.")
                    else:
                        try:
                            set_password(user["username"], new)
                            st.success("Đã đổi mật khẩu.")
                        except ValueError as exc:
                            st.error(str(exc))
        if st.button("Đăng xuất", width="stretch"):
            logout()
