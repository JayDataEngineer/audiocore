#!/usr/bin/env bash
# Point git at scripts/hooks so the gitleaks pre-push hook is active.
# Idempotent — safe to re-run after a fresh clone or a `git config --unset`.
#
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$ROOT/scripts/hooks"

if [[ ! -d "$HOOKS_DIR" ]]; then
    echo "✗ $HOOKS_DIR not found — run this from the repo root." >&2
    exit 1
fi

# Make every hook executable (in case they lost the +x bit on a filesystem
# that doesn't preserve it).
chmod +x "$HOOKS_DIR"/* 2>/dev/null || true

git config core.hooksPath "$HOOKS_DIR"
echo "✓ core.hooksPath = $HOOKS_DIR"
echo "✓ active hooks:"
ls -1 "$HOOKS_DIR" | sed 's/^/    - /'

if command -v gitleaks >/dev/null 2>&1; then
    echo "✓ gitleaks: $(gitleaks version 2>&1)"
else
    cat >&2 <<'MSG'
⚠  gitleaks is not installed. The pre-push hook will warn and skip scans
   until you install it:
     brew install gitleaks
     sudo apt install gitleaks
     https://github.com/gitleaks/gitleaks/releases
MSG
fi

cat <<'MSG'

  To enforce scans on machines without gitleaks installed, set in your shell:
     export AUDIOCORE_REQUIRE_GITLEAKS=1
MSG
