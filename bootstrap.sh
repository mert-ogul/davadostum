#!/usr/bin/env bash
set -e

# Always operate relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow caller to override chosen interpreter
if [[ -n "$PYTHON_BIN" ]]; then
  PY_BIN="$PYTHON_BIN"
else
  # Search for preferred interpreters in descending order of support
  for cand in python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      PY_BIN="$(command -v "$cand")"
      break
    fi
  done
fi

if [[ -z "$PY_BIN" ]]; then
  echo "Error: Could not find a supported Python 3 interpreter (3.10/3.11)." >&2
  exit 1
fi

# Abort if Python major.minor >= 3.12 (many deps lack wheels yet)
PY_MAJOR_MINOR="$($PY_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_MAJOR_MINOR" in
  3.10|3.11) ;;  # supported
  *)
    echo "Detected Python $PY_MAJOR_MINOR — some dependencies lack wheels. Please install Python 3.11 or 3.10 and re-run, e.g. 'brew install python@3.11'." >&2
    exit 1
    ;;
esac

echo "Using Python interpreter: $PY_BIN"

$PY_BIN -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"

# Ensure venv's default python symlinks point to chosen interpreter
ln -sf "$(basename $PY_BIN)" .venv/bin/python
ln -sf "$(basename $PY_BIN)" .venv/bin/python3

# Create necessary directories
mkdir -p data models

# Download embedding model (sentence-transformers will auto-download on first use)
echo "Note: Embedding model will be automatically downloaded on first use"
echo "Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Download Mistral model (optional - users can download manually)
echo "Downloading Mistral model (this may take a while)..."
until huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --include "mistral-7b-instruct-v0.2.Q4_0.gguf" \
  --resume-download \
  --local-dir models; do
  echo "Download interrupted, retrying in 30s ..."
  sleep 30
done

echo "Bootstrap complete ✔"
echo ""
echo "Next steps:"
echo "1. Install Yargı MCP (required dependency):"
echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "   uvx yargi-mcp --help"
echo ""
echo "2. Start MCP server:"
echo "   uvicorn 'yargi_mcp.main:app.http_app' --factory --host 127.0.0.1 --port 3333"
echo ""
echo "3. Run scraper:"
echo "   python -m legalrag.mcp_scraper"
echo ""
echo "4. Run search:"
echo "   python -m legalrag.cli 'your case description'"
echo ""
echo "For more information about Yargı MCP:"
echo "https://github.com/saidsurucu/yargi-mcp"
