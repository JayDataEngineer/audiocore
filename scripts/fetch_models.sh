#!/usr/bin/env bash
# fetch_models.sh — download every file in models/manifest.json into a local
# models directory, then run any family-specific converter (convert_acestep,
# convert_qwen3tts) that the manifest asks for.
#
# Pure bash + curl. No Python, no huggingface-cli dependency. SHA256 is
# verified when a file's `source.sha256` field is present in the manifest.
#
# Usage:
#   scripts/fetch_models.sh                       # fetch every variant
#   scripts/fetch_models.sh ace_step              # one family
#   scripts/fetch_models.sh ace_step ace-step-1.5-turbo   # one variant
#   scripts/fetch_models.sh --list                # print the mode matrix
#   scripts/fetch_models.sh --dry-run             # show what would be fetched
#
# Environment:
#   AUDIOCORE_MODELS_DIR  where to put files (default: ./weights/)
#   AUDIOCORE_BUILD_DIR   where convert_* binaries live (default: ./build/bin)
#   HF_TOKEN              optional auth for gated repos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MANIFEST="$PROJECT_DIR/models/manifest.json"
MODELS_DIR="${AUDIOCORE_MODELS_DIR:-$PROJECT_DIR/weights}"
BUILD_DIR="${AUDIOCORE_BUILD_DIR:-$PROJECT_DIR/build}"
DRY_RUN=0

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required (apt-get install jq / brew install jq)" >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 2
fi
if [ ! -f "$MANIFEST" ]; then
    echo "error: manifest not found at $MANIFEST" >&2
    exit 2
fi

# ── --list: machine-friendly dump of the mode matrix ──────────────────────
if [ "${1:-}" = "--list" ]; then
    jq -r '
      .families | to_entries[] |
      "=== \(.key) — \(.value.display) ===\n" +
      ([.value.modes | to_entries[] |
        "  \(.key)\t\(.value.status)\t\(.value.notes)"] | join("\n")) + "\n"
    ' "$MANIFEST"
    exit 0
fi

# ── filter args ────────────────────────────────────────────────────────────
FILTER_FAMILY=""
FILTER_VARIANT=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *)
            if [ -z "$FILTER_FAMILY" ]; then FILTER_FAMILY="$arg"
            elif [ -z "$FILTER_VARIANT" ]; then FILTER_VARIANT="$arg"
            else echo "too many args: $arg" >&2; exit 2
            fi
            ;;
    esac
done

# ── helpers ────────────────────────────────────────────────────────────────
build_hf_url() {
    # Args: repo revision filename
    local repo="$1" rev="$2" filename="$3"
    echo "https://huggingface.co/${repo}/resolve/${rev}/${filename}"
}

verify_sha256() {
    # Args: expected_hex file_path
    local expected="$1" path="$2"
    local actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$path" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$path" | awk '{print $1}')"
    else
        echo "  warn: neither sha256sum nor shasum present; skipping verify" >&2
        return 0
    fi
    if [ "$actual" != "$expected" ]; then
        echo "  error: sha256 mismatch for $path" >&2
        echo "    expected: $expected" >&2
        echo "    actual:   $actual" >&2
        return 1
    fi
    echo "  ok: sha256 verified"
}

download_one() {
    # Args: family variant dest_dir filename source_obj
    local family="$1" variant="$2" dest_dir="$3" filename="$4" source_json="$5"
    local provider repo rev hf_filename
    provider="$(jq -r '.provider'        <<<"$source_json")"
    repo="$(jq -r '.repo'               <<<"$source_json")"
    rev="$(jq -r '.revision'            <<<"$source_json")"
    hf_filename="$(jq -r '.filename // .subpath // ""' <<<"$source_json")"
    # HF: a `subpath` means we have to enumerate the safetensors in the dir.
    # The C++ converter is the right place to enumerate — fetch_models hands
    # the repo path off to it directly.
    local sha256 subpath_is_set
    sha256="$(jq -r '.sha256 // ""' <<<"$source_json")"
    subpath_is_set="$(jq -r 'has("subpath")' <<<"$source_json")"

    local dest="$dest_dir/$filename"
    mkdir -p "$dest_dir"

    case "$provider" in
        huggingface)
            # `filename` in the manifest means a single resolvable file. `subpath`
            # means a directory the converter scrapes itself — triggers git clone.
            if [ -n "$hf_filename" ] && [[ "$hf_filename" != */ ]] && [ "$subpath_is_set" != "true" ]; then
                local url; url="$(build_hf_url "$repo" "$rev" "$hf_filename")"
                if [ -f "$dest" ]; then
                    echo "  have: $dest"
                else
                    if [ $DRY_RUN -eq 1 ]; then
                        echo "  would: curl -L $url -o $dest"
                    else
                        echo "  fetch: $url → $dest"
                        local auth=()
                        if [ -n "${HF_TOKEN:-}" ]; then
                            auth=(--header "Authorization: Bearer $HF_TOKEN")
                        fi
                        curl -fL "${auth[@]}" "$url" -o "$dest.tmp" && mv "$dest.tmp" "$dest"
                    fi
                fi
                if [ -n "$sha256" ] && [ -f "$dest" ]; then
                    verify_sha256 "$sha256" "$dest" || return 1
                fi
            else
                # Directory case: clone the HF repo via git-LFS pointer flow.
                # Cheaper than reproducing HF's tree API here.
                local repo_dir="$dest_dir/$repo"
                if [ -d "$repo_dir/.git" ]; then
                    echo "  have: $repo_dir (git)"
                else
                    mkdir -p "$(dirname "$repo_dir")"
                    if [ $DRY_RUN -eq 1 ]; then
                        echo "  would: git clone https://huggingface.co/$repo $repo_dir"
                    else
                        echo "  clone: https://huggingface.co/$repo → $repo_dir"
                        git clone "https://huggingface.co/${repo}" "$repo_dir"
                    fi
                fi
            fi
            ;;
        *)
            echo "  error: unknown provider '$provider' for $family/$variant/$filename" >&2
            return 1
            ;;
    esac

    # Optional post_process: invoke a converter binary.
    local post; post="$(jq -r '.post_process // .convert_with // ""' <<<"$source_json")"
    if [ -n "$post" ]; then
        # We only know about two converters today.
        case "$post" in
            convert_acestep|convert_qwen3tts)
                local bin="$BUILD_DIR/$post"
                if [ ! -x "$bin" ]; then
                    echo "  warn: $bin not found; skipping conversion (build it with cmake --build build --target $post)" >&2
                    return 0
                fi
                # convert_acestep rewrites in place; convert_qwen3tts takes a dir.
                if [ "$post" = "convert_acestep" ]; then
                    echo "  convert (in-place): $bin --in $dest"
                    [ $DRY_RUN -eq 1 ] || "$bin" --in "$dest"
                else
                    echo "  convert: $bin $dest_dir/$repo --outdir $dest_dir"
                    [ $DRY_RUN -eq 1 ] || "$bin" "$dest_dir/$repo" --outdir "$dest_dir"
                fi
                ;;
            *)
                echo "  warn: unknown converter '$post'; skipping" >&2
                ;;
        esac
    fi
}

# ── main loop ──────────────────────────────────────────────────────────────
families_json="$(jq -c '.families | keys[]' "$MANIFEST")"
status=0
while IFS= read -r family_quoted; do
    family="${family_quoted//\"/}"
    if [ -n "$FILTER_FAMILY" ] && [ "$family" != "$FILTER_FAMILY" ]; then continue; fi

    echo "── $family ────────────────────────────────────────────────────────"
    variants_json="$(jq -c --arg f "$family" '.families[$f].variants | keys[]' "$MANIFEST")"
    while IFS= read -r variant_quoted; do
        variant="${variant_quoted//\"/}"
        if [ -n "$FILTER_VARIANT" ] && [ "$variant" != "$FILTER_VARIANT" ]; then continue; fi

        display="$(jq -r --arg f "$family" --arg v "$variant" '.families[$f].variants[$v].display' "$MANIFEST")"
        echo "  ▸ $variant — $display"
        dest_dir="$MODELS_DIR/$family/$variant"
        files_json="$(jq -c --arg f "$family" --arg v "$variant" '.families[$f].variants[$v].files[]' "$MANIFEST")"
        while IFS= read -r file_obj; do
            filename="$(jq -r '.filename' <<<"$file_obj")"
            source_json="$(jq -c '.source' <<<"$file_obj")"
            download_one "$family" "$variant" "$dest_dir" "$filename" "$source_json" || status=1
        done <<<"$files_json"
    done <<<"$variants_json"
done <<<"$families_json"

if [ $status -ne 0 ]; then
    echo "fetch_models: one or more downloads failed" >&2
    exit $status
fi

echo "done. Drop a server.json entry referencing $MODELS_DIR and start audiocore_server."
