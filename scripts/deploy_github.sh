#!/usr/bin/env bash
# Prepara e envia PepMem-AI para GitHub (repo público).
set -euo pipefail
cd "$(dirname "$0")/.."

GITHUB_USER="${1:-}"
REPO_NAME="${2:-PepMem-AI}"

if [[ -z "$GITHUB_USER" ]]; then
  echo "Uso: ./scripts/deploy_github.sh SEU_USUARIO_GITHUB [nome-do-repo]"
  echo "Ex.: ./scripts/deploy_github.sh gabriel-motta PepMem-AI"
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git branch -M main
fi

git add -A
git status --short

if git diff --cached --quiet; then
  echo "Nada novo para commitar."
else
  git commit -m "$(cat <<'EOF'
Publica PepMem-AI PoC com dashboard, SHAP e datasets processados.

Inclui pipeline de bioprospecção peptídeo–membrana para deploy no Hugging Face Spaces.
EOF
)"
fi

REMOTE="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo ""
echo "Próximo passo (crie o repo vazio em github.com/new primeiro):"
echo "  git push -u origin main"
echo ""
echo "URL do repo: https://github.com/${GITHUB_USER}/${REPO_NAME}"
