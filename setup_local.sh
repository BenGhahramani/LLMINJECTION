#!/usr/bin/env bash
# setup_local.sh — one-shot setup for running the LLM Injection Test Bench
# with a local Ollama model (no OpenAI API key required).
#
# Usage:
#   chmod +x setup_local.sh
#   ./setup_local.sh              # uses default model (qwen2.5:7b)
#   ./setup_local.sh llama3.1:8b  # or any other Ollama model that supports tool-calling
#
# Supported OS: Linux, macOS
# Requirements: curl, Python 3.10+

set -euo pipefail

MODEL="${1:-qwen2.5:7b}"
OLLAMA_BASE_URL="http://localhost:11434/v1"
ENV_FILE=".env"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
warn()  { echo "[WARN]  $*"; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' is required but not found. Please install it."
}

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------

info "Checking prerequisites..."
require_cmd curl
require_cmd python3

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 || ( "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ) ]]; then
    die "Python 3.10+ required (found $PYTHON_VERSION)"
fi
ok "Python $PYTHON_VERSION"

# ---------------------------------------------------------------------------
# 2. Install Ollama
# ---------------------------------------------------------------------------

if command -v ollama &>/dev/null; then
    ok "Ollama already installed ($(ollama --version 2>/dev/null | head -1))"
else
    info "Installing Ollama..."
    case "$(uname -s)" in
        Linux)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        Darwin)
            if command -v brew &>/dev/null; then
                brew install ollama
            else
                die "On macOS, install Ollama manually from https://ollama.com/download or install Homebrew first."
            fi
            ;;
        *)
            die "Unsupported OS '$(uname -s)'. Install Ollama manually from https://ollama.com/download"
            ;;
    esac
    ok "Ollama installed"
fi

# ---------------------------------------------------------------------------
# 3. Start Ollama service (if not already running)
# ---------------------------------------------------------------------------

if curl -sf http://localhost:11434 &>/dev/null; then
    ok "Ollama service already running"
else
    info "Starting Ollama service in background..."
    ollama serve &>/tmp/ollama.log &
    OLLAMA_PID=$!
    # Wait up to 15 seconds for it to become ready
    for i in $(seq 1 15); do
        sleep 1
        if curl -sf http://localhost:11434 &>/dev/null; then
            ok "Ollama service started (PID $OLLAMA_PID)"
            break
        fi
        if [[ $i -eq 15 ]]; then
            die "Ollama service did not start in time. Check /tmp/ollama.log"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 4. Pull the model
# ---------------------------------------------------------------------------

info "Pulling model '$MODEL' (this may take a few minutes on first run)..."
ollama pull "$MODEL"
ok "Model '$MODEL' ready"

# ---------------------------------------------------------------------------
# 5. Python virtual environment + dependencies
# ---------------------------------------------------------------------------

if [[ ! -d "venv" ]]; then
    info "Creating virtual environment..."
    python3 -m venv venv
    ok "venv created"
else
    ok "venv already exists"
fi

info "Installing Python dependencies..."
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
ok "Dependencies installed"

# ---------------------------------------------------------------------------
# 6. Write .env
# ---------------------------------------------------------------------------

if [[ -f "$ENV_FILE" ]]; then
    # Update or insert LLM_BASE_URL and LLM_MODEL, preserve everything else.
    # Remove existing entries for these two keys first, then append.
    TMP=$(mktemp)
    grep -v '^LLM_BASE_URL=' "$ENV_FILE" | grep -v '^LLM_MODEL=' > "$TMP" || true
    printf '\nLLM_BASE_URL=%s\nLLM_MODEL=%s\n' "$OLLAMA_BASE_URL" "$MODEL" >> "$TMP"
    mv "$TMP" "$ENV_FILE"
    ok "Updated $ENV_FILE with local LLM settings"
else
    cat > "$ENV_FILE" <<EOF
# LLM backend — local Ollama (set by setup_local.sh)
LLM_BASE_URL=${OLLAMA_BASE_URL}
LLM_MODEL=${MODEL}

# Leave OPENAI_API_KEY blank when using a local model.
# OPENAI_API_KEY=
EOF
    ok "Created $ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "======================================================"
echo "  Setup complete!"
echo "======================================================"
echo ""
echo "  Model : $MODEL"
echo "  Ollama: $OLLAMA_BASE_URL"
echo ""
echo "  Start the app:"
echo "    source venv/bin/activate"
echo "    uvicorn backend.app:app --reload"
echo ""
echo "  Then open http://localhost:8000 in your browser."
echo ""
echo "  To change model, re-run:"
echo "    ./setup_local.sh <model-name>"
echo ""
echo "  Browse available models: https://ollama.com/library"
echo "  Good tool-calling models: qwen2.5:7b, llama3.1:8b, mistral:7b-instruct"
echo "======================================================"
