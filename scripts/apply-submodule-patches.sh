#!/usr/bin/env bash
# Re-apply the patches under patches/<submodule>/ to the corresponding
# submodule after a fresh checkout. Idempotent: skips patches that are
# already applied. Safe to run repeatedly.
#
# Typical use:
#   git submodule update --init --recursive
#   ./scripts/apply-submodule-patches.sh
#
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

declare -A SUBMODULE_OF_DIR=(
    ["llama.cpp"]="third_party/llama.cpp"
)

for patch_dir in "$ROOT"/patches/*/; do
    [ -d "$patch_dir" ] || continue
    name=$(basename "$patch_dir")
    sub="${SUBMODULE_OF_DIR[$name]:-}"
    if [ -z "$sub" ]; then
        echo "apply-submodule-patches: no submodule mapping for $name — skipping" >&2
        continue
    fi
    target="$ROOT/$sub"
    if [ ! -d "$target/.git" ] && [ ! -f "$target/.git" ]; then
        echo "apply-submodule-patches: $target is not a checked-out submodule — skipping" >&2
        continue
    fi

    echo "== applying patches for $name =="
    # Run inside the submodule so paths in the patch match.
    shopt -s nullglob
    for patch in "$patch_dir"*.patch; do
        if git -C "$target" apply --check "$patch" >/dev/null 2>&1; then
            git -C "$target" apply "$patch"
            echo "  + $(basename "$patch")"
        else
            # Already applied, or context shifted. Detect "already applied"
            # by reverse-applying: if reverse applies cleanly, the patch is
            # already in place and we skip silently.
            if git -C "$target" apply --check --reverse "$patch" >/dev/null 2>&1; then
                echo "  = $(basename "$patch") (already applied)"
            else
                echo "  ! $(basename "$patch") did not apply cleanly — manual review needed" >&2
                git -C "$target" apply --stat "$patch" >&2 || true
                exit 1
            fi
        fi
    done
done
