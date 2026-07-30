"""Gate de login do dashboard PepMem-AI.

Credenciais via ``st.secrets["auth"]["users"]`` (mapa usuário → senha) ou
variáveis de ambiente ``AUTH_USER`` / ``AUTH_PASSWORD``. Bypass local explícito:
``PEPMEM_AUTH_DISABLED=1``.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

_SESSION_USER = "pepmem_auth_user"
_SESSION_OK = "pepmem_authenticated"


def auth_disabled() -> bool:
    """True se o gate estiver desligado (só via env explícito)."""
    return os.environ.get("PEPMEM_AUTH_DISABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _users_from_secrets() -> dict[str, str]:
    """Lê mapa usuário→senha de st.secrets, se existir."""
    try:
        auth = st.secrets.get("auth", None)
    except Exception:
        return {}
    if auth is None:
        return {}
    # Streamlit AttrDict / dict
    users = auth.get("users") if hasattr(auth, "get") else None
    if users is None:
        return {}
    out: dict[str, str] = {}
    try:
        items = users.items()
    except Exception:
        return {}
    for k, v in items:
        if k is None or v is None:
            continue
        out[str(k).strip()] = str(v)
    return out


def _users_from_env() -> dict[str, str]:
    user = os.environ.get("AUTH_USER", "").strip()
    password = os.environ.get("AUTH_PASSWORD", "")
    if user and password:
        return {user: password}
    return {}


def load_users() -> dict[str, str]:
    """Usuários configurados (secrets têm prioridade; env complementa se secrets vazio)."""
    users = _users_from_secrets()
    if users:
        return users
    return _users_from_env()


def is_authenticated() -> bool:
    if auth_disabled():
        return True
    return bool(st.session_state.get(_SESSION_OK))


def current_user() -> str | None:
    if auth_disabled():
        return os.environ.get("AUTH_USER") or "dev"
    return st.session_state.get(_SESSION_USER)


def login(username: str, password: str) -> bool:
    """Valida credenciais e grava sessão. Retorna True se ok."""
    users = load_users()
    user = (username or "").strip()
    pwd = password or ""
    expected = users.get(user)
    if expected is None:
        return False
    if not hmac.compare_digest(str(expected), str(pwd)):
        return False
    st.session_state[_SESSION_OK] = True
    st.session_state[_SESSION_USER] = user
    return True


def logout() -> None:
    st.session_state.pop(_SESSION_OK, None)
    st.session_state.pop(_SESSION_USER, None)


def _render_login_form(error: str | None = None) -> None:
    """Tela centrada de login (visual PepMem)."""
    st.markdown(
        """
        <div class="pepmem-login-wrap">
          <div class="pepmem-login-card">
            <div class="pepmem-login-brand">InovAI Lab · UFRN</div>
            <div class="pepmem-login-title">PepMem-AI</div>
            <div class="pepmem-login-sub">Acesso restrito ao dashboard de priorização</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.form("pepmem_login_form", clear_on_submit=False):
            user = st.text_input("Usuário", autocomplete="username")
            password = st.text_input("Senha", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")
        if error:
            st.error(error)
        if submitted:
            if not load_users():
                st.error(
                    "Nenhuma credencial configurada. Defina "
                    "`st.secrets['auth']['users']` ou AUTH_USER/AUTH_PASSWORD, "
                    "ou use PEPMEM_AUTH_DISABLED=1 só em desenvolvimento local."
                )
            elif login(user, password):
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")


def require_login() -> None:
    """Bloqueia o restante do app até autenticação bem-sucedida."""
    if auth_disabled():
        return
    if is_authenticated():
        return

    users = load_users()
    if not users:
        st.markdown("### PepMem-AI · Login")
        st.warning(
            "Autenticação ativa, mas **nenhuma credencial** foi configurada.\n\n"
            "Configure `.streamlit/secrets.toml` com `[auth.users]` "
            "ou as variáveis `AUTH_USER` / `AUTH_PASSWORD`.\n\n"
            "Para desenvolvimento local sem senha: "
            "`export PEPMEM_AUTH_DISABLED=1`."
        )
        st.stop()

    _render_login_form()
    st.stop()


def render_sidebar_auth() -> None:
    """Mostra usuário logado e botão Sair (chamar dentro de ``st.sidebar``)."""
    if auth_disabled():
        st.caption("Auth desligada (PEPMEM_AUTH_DISABLED)")
        return
    user = current_user() or "—"
    st.caption(f"Logado como **{user}**")
    if st.button("Sair", use_container_width=True, key="pepmem_logout"):
        logout()
        st.rerun()
