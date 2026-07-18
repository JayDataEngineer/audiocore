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
PYTHON_ENGINE_MODE=0
POS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --python-engine) PYTHON_ENGINE_MODE=1 ;;
        --*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *)
            POS_ARGS+=("$arg")
            ;;
    esac
done
# Positional args: family and variant (scoped filtering)
if [ ${#POS_ARGS[@]} -ge 1 ]; then FILTER_FAMILY="${POS_ARGS[0]}"; fi
if [ ${#POS_ARGS[@]} -ge 2 ]; then FILTER_VARIANT="${POS_ARGS[1]}"; fi
if [ ${#POS_ARGS[@]} -ge 3 ]; then echo "too many args: ${POS_ARGS[2]}" >&2; exit 2; fi

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
        # The manifest may write either "convert_acestep" or
        # "convert_acestep --in <file>"; strip args for the dispatch match
        # and substitute <file> in the remaining args afterwards.
        local post_name="${post%% *}"
        local post_args="${post#"$post_name"}"
        # Substitute <file> placeholder with the destination file path.
        post_args="${post_args//<file>/$dest}"
        # Trim leading whitespace.
        post_args="${post_args#"${post_args%%[![:space:]]*}"}"
        case "$post_name" in
            convert_acestep|convert_qwen3tts)
                local bin="$BUILD_DIR/$post_name"
                if [ ! -x "$bin" ]; then
                    echo "  warn: $bin not found; skipping conversion (build it with cmake --build build --target $post_name)" >&2
                    return 0
                fi
                if [ "$post_name" = "convert_acestep" ]; then
                    # In-place conversion: the args typically carry --in <file>.
                    # If no args, default to --in $dest for backward compat.
                    if [ -z "$post_args" ]; then
                        post_args="--in $dest"
                    fi
                    echo "  convert (in-place): $bin $post_args"
                    [ $DRY_RUN -eq 1 ] || "$bin" $post_args
                else
                    # convert_qwen3tts takes the upstream repo dir + --outdir.
                    # Manifest may pass explicit args; otherwise default.
                    if [ -z "$post_args" ]; then
                        echo "  convert: $bin $dest_dir/$repo --outdir $dest_dir"
                        [ $DRY_RUN -eq 1 ] || "$bin" "$dest_dir/$repo" --outdir "$dest_dir"
                    else
                        echo "  convert: $bin $post_args"
                        [ $DRY_RUN -eq 1 ] || "$bin" $post_args
                    fi
                fi
                ;;
            convert_ecapa)
                local bin="$BUILD_DIR/$post_name"
                if [ ! -x "$bin" ]; then
                    echo "  warn: $bin not found; skipping conversion (build it with cmake --build build --target $post_name)" >&2
                    return 0
                fi
                if [ -z "$post_args" ]; then
                    post_args="$dest_dir/$repo --outdir $dest_dir"
                fi
                echo "  convert: $bin $post_args"
                [ $DRY_RUN -eq 1 ] || "$bin" $post_args
                ;;
            convert_moss_tts)
                # Python converter: extracts Qwen3 backbone from MOSS
                # safetensors via llama.cpp, then writes a sidecar GGUF
                # carrying audio embeddings + LM heads + codec tensors.
                local script="$PROJECT_DIR/tools/convert_moss_tts.py"
                if [ ! -f "$script" ]; then
                    echo "  warn: $script not found; skipping conversion" >&2
                    return 0
                fi
                # Upstream model + codec repos were cloned to
                # $dest_dir/$repo (nested: e.g. $dest_dir/OpenMOSS-Team/...).
                # The codec lives in a sibling clone of MOSS-Audio-Tokenizer.
                local model_repo="$dest_dir/$repo"
                local codec_repo=""
                local cand
                # Search two levels deep to find the codec clone.
                while IFS= read -r -d '' cand; do
                    case "$(basename "$cand")" in
                        *Audio-Tokenizer*|*audio-tokenizer*|*codec*)
                            codec_repo="$cand"; break ;;
                    esac
                done < <(find "$dest_dir" -maxdepth 3 -type d -printf '%p\0' 2>/dev/null)
                # Output GGUF is named after the variant (e.g.
                # moss-tts-f16.gguf, moss-sfx-f16.gguf, moss-voicegen-q8_0.gguf).
                local base_name="$variant"
                # Strip any explicit -f16 / -q4_k_m / -q8_0 suffix from the
                # variant name — the dtype/quant suffix is added back below
                # based on --backbone-dtype and the optional quantize pass.
                base_name="${base_name%-f16}"
                base_name="${base_name%-q4-k-m}"
                base_name="${base_name%-q8_0}"
                local dtype="f16"
                local out_gguf="$dest_dir/${variant}.gguf"
                local out_sidecar="$dest_dir/${variant}.extras.gguf"
                if [ -f "$out_gguf" ] && [ -f "$out_sidecar" ]; then
                    echo "  have: $out_gguf (+ sidecar)"
                else
                    local codec_args=()
                    if [ -n "$codec_repo" ] && [ -d "$codec_repo" ]; then
                        codec_args=(--codec "$codec_repo")
                    fi
                    echo "  convert: python3 $(basename "$script") --moss-tts $model_repo ${codec_args[*]:-} --output $out_gguf"
                    if [ $DRY_RUN -ne 1 ]; then
                        python3 "$script" \
                            --moss-tts "$model_repo" \
                            "${codec_args[@]}" \
                            --output "$out_gguf" \
                            --backbone-dtype "$dtype" \
                            --scratch-dir "$dest_dir/scratch" || return 1
                    fi
                fi
                # Optional quantization pass.
                local quant; quant="$(jq -r '.post_quantize // ""' <<<"$source_json")"
                if [ -n "$quant" ]; then
                    # llama-quantize may live at $BUILD_DIR/llama-quantize
                    # (legacy) or $BUILD_DIR/bin/llama-quantize (current
                    # llama.cpp layout).
                    local qbin=""
                    for cand in \
                        "$BUILD_DIR/llama-quantize" \
                        "$BUILD_DIR/bin/llama-quantize"; do
                        if [ -x "$cand" ]; then qbin="$cand"; break; fi
                    done
                    if [ -n "$qbin" ] && [ -f "$out_gguf" ]; then
                        # Name the quantized output with the variant's own
                        # basename + lowercase quant suffix.
                        local qout="$dest_dir/${base_name}-${quant,,}.gguf"
                        echo "  quantize: $qbin $out_gguf $qout $quant"
                        if [ $DRY_RUN -ne 1 ]; then
                            "$qbin" "$out_gguf" "$qout" "$quant" || return 1
                            # Remove the F16 intermediate so the directory
                            # scanner picks the quantized file. Without this,
                            # sorted() picks "moss-tts-q4-k-m.gguf" (F16)
                            # before "moss-tts-q4_k_m.gguf" (quantized).
                            rm -f "$out_gguf"
                        fi
                    elif [ -z "$qbin" ]; then
                        echo "  warn: llama-quantize not found (looked in $BUILD_DIR and $BUILD_DIR/bin)" >&2
                        echo "         build with: cmake -S . -B build -DENGINE_BUILD_LLAMA_QUANTIZE=ON && cmake --build build --target llama-quantize" >&2
                        echo "         F16 retained" >&2
                    fi
                fi
                ;;
            *)
                echo "  warn: unknown converter '$post'; skipping" >&2
                ;;
        esac
    fi

    # Optional sidecar: download a companion .extras.gguf alongside the main file.
    local sidecar; sidecar="$(jq -r '.sidecar // ""' <<<"$source_json")"
    if [ -n "$sidecar" ]; then
        local sidecar_url; sidecar_url="$(build_hf_url "$repo" "$rev" "$sidecar")"
        local sidecar_dest="$dest_dir/$sidecar"
        if [ -f "$sidecar_dest" ]; then
            echo "  have: $sidecar_dest"
        else
            if [ $DRY_RUN -eq 1 ]; then
                echo "  would: curl -L $sidecar_url -o $sidecar_dest"
            else
                echo "  fetch: $sidecar_url → $sidecar_dest"
                local auth=()
                if [ -n "${HF_TOKEN:-}" ]; then
                    auth=(--header "Authorization: Bearer $HF_TOKEN")
                fi
                curl -fL "${auth[@]}" "$sidecar_url" -o "$sidecar_dest.tmp" && mv "$sidecar_dest.tmp" "$sidecar_dest"
            fi
        fi
    fi
}

# ── Python engine HF source provisioning ──────────────────────────────────
# The qwen3_tts Python engine consumes HF source dirs directly (no GGUF
# conversion). snapshot_download lays them down in either flat or models--
# cache layout. This is the IAC path for the Python engine variants
# declared under families.<f>.python_engine.variants in the manifest.
fetch_python_engine() {
    # Args: [family] [variant]   (both optional; empty = all)
    local want_family="${1:-}"
    local want_variant="${2:-}"
    local hf_root="${AUDIOCORE_PYTHON_HF_ROOT:-/mnt/data/models/hf_cache}"
    local fams
    if [ -n "$want_family" ]; then
        fams="$(jq -c --arg f "$want_family" 'if (.families[$f].python_engine) then [$f] else [] end' "$MANIFEST")"
    else
        fams="$(jq -c '[.families | to_entries[] | select(.value.python_engine) | .key]' "$MANIFEST")"
    fi
    if [ "$fams" = "[]" ]; then
        echo "fetch_python_engine: no families with a python_engine section matched" >&2
        return 1
    fi
    echo "── python_engine provisioning (hf_root=$hf_root) ───────────────────"
    command -v python3 >/dev/null 2>&1 || { echo "error: python3 required for python_engine provisioning" >&2; return 2; }
    local fam variant repo rev display
    for fam in $(jq -r '.[]' <<<"$fams"); do
        echo "  family: $fam"
        local variants_json
        if [ -n "$want_variant" ]; then
            # Single variant scope: emit only if it exists under this family.
            variants_json="$(jq -c --arg f "$fam" --arg v "$want_variant" \
                'if (.families[$f].python_engine.variants[$v]) then "\($v)" else empty end' "$MANIFEST")"
        else
            variants_json="$(jq -c --arg f "$fam" '.families[$f].python_engine.variants | keys[]' "$MANIFEST")"
        fi
        while IFS= read -r variant_quoted; do
            [ -z "$variant_quoted" ] && continue
            variant="${variant_quoted//\"/}"
            repo="$(jq -r --arg f "$fam" --arg v "$variant" \
                '.families[$f].python_engine.variants[$v].hf_repo' "$MANIFEST")"
            rev="$(jq -r --arg f "$fam" --arg v "$variant" \
                '.families[$f].python_engine.variants[$v].hf_revision // "main"' "$MANIFEST")"
            display="$(jq -r --arg f "$fam" --arg v "$variant" \
                '.families[$f].python_engine.variants[$v].display' "$MANIFEST")"
            echo "    ▸ $variant — $display"
            echo "      hf_repo: $repo @$rev → $hf_root"
            if [ "$DRY_RUN" = "1" ]; then
                echo "      (dry-run) would call snapshot_download"
                continue
            fi
            HF_TOKEN="${HF_TOKEN:-}" python3 - "$hf_root" "$repo" "$rev" <<'PYEOF' || return 1
import os, sys
hf_root, repo, rev = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"      error: huggingface_hub not installed: {e}", file=sys.stderr)
    sys.exit(2)
os.makedirs(hf_root, exist_ok=True)
path = snapshot_download(
    repo_id=repo, revision=rev, cache_dir=hf_root,
    local_files_only=False,
)
print(f"      → {path}")
PYEOF
        done <<<"$variants_json"
    done
}

# ── main loop ──────────────────────────────────────────────────────────────
# --python-engine: alternative mode that runs only the HF source download
# path for the Python engine (no GGUF conversion). Pass an optional family
# name to scope it. Runs AFTER fetch_python_engine is defined above.
if [ "$PYTHON_ENGINE_MODE" = "1" ]; then
    fetch_python_engine "$FILTER_FAMILY" "$FILTER_VARIANT"
    exit $?
fi

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
            # Merge file-level post_process / post_quantize down into the
            # source object so download_one's single-arg lookup works whether
            # the manifest used the legacy `source.convert_with` form or the
            # newer file-level `post_process` form.
            source_json="$(jq -c \
                --argjson pp "$(jq -r '.post_process // .source.convert_with // ""' <<<"$file_obj" | jq -R '{post_process: .}')" \
                --argjson pq "$(jq -r '.post_quantize // ""' <<<"$file_obj" | jq -R '{post_quantize: .}')" \
                '.source + $pp + $pq' <<<"$file_obj")"
            download_one "$family" "$variant" "$dest_dir" "$filename" "$source_json" || status=1
        done <<<"$files_json"
    done <<<"$variants_json"
done <<<"$families_json"

if [ $status -ne 0 ]; then
    echo "fetch_models: one or more downloads failed" >&2
    exit $status
fi

echo "done. Drop a server.json entry referencing $MODELS_DIR and start audiocore_server."
