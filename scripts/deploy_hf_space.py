#!/usr/bin/env python3
"""Build staging folder and deploy PepMem-AI to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".deploy_hf"

COPY_DIRS = ("pepmem", "dashboard", "api", "scripts", ".streamlit")
COPY_FILES = ("requirements-space.txt",)


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))}


def build_stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    processed_src = ROOT / "data" / "processed"
    if not processed_src.exists():
        raise SystemExit("data/processed/ ausente. Execute scripts/run_pipeline.py primeiro.")

    for name in COPY_DIRS:
        src = ROOT / name
        if not src.exists():
            raise SystemExit(f"Diretório obrigatório ausente: {name}")
        shutil.copytree(src, STAGE / name, ignore=_ignore)

    shutil.copytree(processed_src, STAGE / "data" / "processed", ignore=_ignore)

    for name in COPY_FILES:
        shutil.copy2(ROOT / name, STAGE / name)

    shutil.copy2(ROOT / "requirements-space.txt", STAGE / "requirements.txt")
    shutil.copy2(ROOT / "deploy" / "README_HF.md", STAGE / "README.md")

    print(f"Staging pronto: {STAGE} ({_dir_size(STAGE):.1f} MB)")
    return STAGE


def _dir_size(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def deploy(username: str, space_name: str, token: str | None) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = f"{username}/{space_name}"
    api = HfApi(token=token)

    who = api.whoami(token=token)
    print(f"Logado como: {who.get('name', who)}")

    stage = build_stage()
    use_docker = True  # HF free/PRO: sdk streamlit removido; Spaces de app usam docker/gradio

    # Reaproveita Space existente (não tenta create_repo — Docker novo exige PRO)
    space_exists = False
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="space", token=token)
        space_exists = True
        print(f"Space existente: {repo_id}")
        _ = info
    except RepositoryNotFoundError:
        space_exists = False

    if not space_exists:
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                token=token,
            )
            print(f"Space criado: {repo_id} (docker)")
        except Exception as exc:
            raise SystemExit(
                "Não foi possível criar o Space (HF pode exigir PRO para Docker).\n"
                f"Detalhe: {exc}\n"
                "Crie manualmente em https://huggingface.co/new-space (Docker) "
                "ou reutilize um Space já existente e rode o deploy de novo."
            ) from exc

    shutil.copy2(ROOT / "Dockerfile", stage / "Dockerfile")
    dockerfile = (stage / "Dockerfile").read_text(encoding="utf-8")
    dockerfile = dockerfile.replace("8501", "7860")
    if "STREAMLIT_SERVER_FILE_WATCHER" not in dockerfile:
        dockerfile = dockerfile.replace(
            "ENV PYTHONPATH=/app",
            "ENV PYTHONPATH=/app\nENV STREAMLIT_SERVER_FILE_WATCHER=none",
        )
    (stage / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    readme = (stage / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("sdk: streamlit", "sdk: docker").replace(
        "app_file: dashboard/app.py\n", ""
    )
    (stage / "README.md").write_text(readme, encoding="utf-8")

    print(f"Enviando para Space {repo_id} (docker)...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message="Deploy PepMem-AI dashboard",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nSpace publicado: {url}")
    print("Aguarde o rebuild no HF (pode levar alguns minutos).")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy PepMem-AI no Hugging Face Spaces")
    parser.add_argument("username", nargs="?", default=os.environ.get("HF_USERNAME"), help="Usuário HF")
    parser.add_argument("--space", default=os.environ.get("HF_SPACE", "pepmem-ai"), help="Nome do Space")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Token HF (ou env HF_TOKEN)")
    parser.add_argument("--build-only", action="store_true", help="Só monta .deploy_hf/, sem upload")
    args = parser.parse_args()

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key == "HF_TOKEN" and not args.token:
                args.token = val
            elif key == "HF_USERNAME" and not args.username:
                args.username = val
            elif key == "HF_SPACE" and args.space == "pepmem-ai":
                args.space = val

    if args.build_only:
        build_stage()
        return

    if not args.token:
        print(
            "Token ausente.\n\n"
            "1. Crie token: https://huggingface.co/settings/tokens (Write)\n"
            "2. Opção A — login:\n"
            "     hf auth login\n"
            "     python scripts/deploy_hf_space.py SEU_USUARIO\n"
            "   Opção B — arquivo .env na raiz do projeto:\n"
            "     HF_USERNAME=seu_usuario\n"
            "     HF_TOKEN=hf_...\n"
            "     python scripts/deploy_hf_space.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    who = api.whoami(token=args.token)
    username = args.username or who.get("name") or who.get("fullname")
    if not username:
        print("Não foi possível detectar usuário HF. Passe como argumento.", file=sys.stderr)
        sys.exit(1)

    deploy(username, args.space, args.token)


if __name__ == "__main__":
    main()
