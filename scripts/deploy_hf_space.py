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

    repo_id = f"{username}/{space_name}"
    api = HfApi(token=token)

    who = api.whoami(token=token)
    print(f"Logado como: {who.get('name', who)}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
        token=token,
    )

    stage = build_stage()
    print(f"Enviando para Space {repo_id}...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message="Deploy PepMem-AI dashboard",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nSpace publicado: {url}")
    print("Aguarde 10–20 min para o build (PyTorch na 1ª vez).")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy PepMem-AI no Hugging Face Spaces")
    parser.add_argument("username", help="Seu usuário Hugging Face (ex.: gabriel-motta)")
    parser.add_argument("--space", default="pepmem-ai", help="Nome do Space (padrão: pepmem-ai)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Token HF (ou env HF_TOKEN)")
    parser.add_argument("--build-only", action="store_true", help="Só monta .deploy_hf/, sem upload")
    args = parser.parse_args()

    if args.build_only:
        build_stage()
        return

    if not args.token:
        print(
            "Token ausente. Faça login:\n"
            "  hf auth login\n"
            "ou export HF_TOKEN=hf_...\n"
            "Depois rode novamente este script.",
            file=sys.stderr,
        )
        sys.exit(1)

    deploy(args.username, args.space, args.token)


if __name__ == "__main__":
    main()
