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
    """Tela de login com cabeçalho PepMem e formulário bem sinalizado."""
    # Esconde a sidebar nesta tela (só faz sentido depois de autenticar)
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"],
          [data-testid="stSidebarCollapsedControl"],
          section[data-testid="stSidebar"] { display: none !important; }
          .block-container { max-width: 560px !important; padding-top: 1.2rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pepmem-login-hero">
          <div class="pepmem-login-hero-brand">INOVAI LAB · UFRN · TITYUS STIGMURUS</div>
          <div class="pepmem-login-hero-title">PepMem-AI</div>
          <div class="pepmem-login-hero-sub">
            Predição de interação peptídeo–membrana · acesso restrito a colaboradores
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pepmem-login-panel">
          <div class="pepmem-login-panel-title">Entrar no dashboard</div>
          <div class="pepmem-login-panel-hint">
            Use o usuário e a senha fornecidos pelo laboratório.
            Digite nos campos abaixo e clique em <strong>Entrar</strong>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("pepmem_login_form", clear_on_submit=False):
        st.markdown("##### 1. Usuário")
        user = st.text_input(
            "Usuário",
            placeholder="Ex.: admin",
            autocomplete="username",
            label_visibility="collapsed",
            help="Identificador combinado com a equipe (ex.: admin, colaborador).",
        )
        st.markdown("##### 2. Senha")
        password = st.text_input(
            "Senha",
            type="password",
            placeholder="Digite sua senha",
            autocomplete="current-password",
            label_visibility="collapsed",
            help="Senha definida nos secrets do app (não compartilhe em canais públicos).",
        )
        st.caption("Os campos acima são obrigatórios. Sem usuário/senha válidos o dashboard não abre.")
        submitted = st.form_submit_button("Entrar no PepMem-AI", use_container_width=True, type="primary")

    if error:
        st.error(error)
    if submitted:
        if not (user or "").strip() or not password:
            st.warning("Preencha **usuário** e **senha** antes de entrar.")
        elif not load_users():
            st.error(
                "Nenhuma credencial configurada no servidor. "
                "Peça ao administrador para definir `auth.users` nos secrets."
            )
        elif login(user, password):
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos. Verifique e tente de novo.")

    st.markdown(
        """
        <div class="pepmem-login-foot">
          Problemas de acesso? Contate o InovAI Lab / responsável pelo projeto.
          Este login apenas restringe o PoC — não é autenticação institucional completa.
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_login() -> None:
    """Bloqueia o restante do app até autenticação bem-sucedida."""
    if auth_disabled():
        return
    if is_authenticated():
        return

    users = load_users()
    if not users:
        st.markdown(
            """
            <style>
              [data-testid="stSidebar"],
              section[data-testid="stSidebar"] { display: none !important; }
            </style>
            <div class="pepmem-login-hero">
              <div class="pepmem-login-hero-brand">INOVAI LAB · UFRN</div>
              <div class="pepmem-login-hero-title">PepMem-AI</div>
              <div class="pepmem-login-hero-sub">Login necessário, mas ainda sem credenciais no servidor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Autenticação ativa, mas **nenhuma credencial** foi configurada.\n\n"
            "Configure `.streamlit/secrets.toml` ou os Secrets do Cloud com:\n\n"
            "```toml\n[auth.users]\nadmin = \"sua-senha\"\n```\n\n"
            "Desenvolvimento local sem senha: `export PEPMEM_AUTH_DISABLED=1`."
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
