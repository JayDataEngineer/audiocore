#!/usr/bin/env bash
# audiocore container entrypoint.
#
# Responsibilities:
#   1. Materialize /etc/audiocore/server.json from the default if the user
#      mounted only /models and not a config.
#   2. Validate the config parses as JSON and that every model path exists.
#   3. exec the server under tini (PID 1 already is tini; this drops the
#      extra shell layer via exec).
#
# Config resolution order:
#   - $AUDIOCORE_CONFIG if it points to an existing file
#   - /etc/audiocore/server.json if it exists (user-mounted)
#   - /etc/audiocore/server.json.default (shipped copy) as a fallback ONLY
#     if AUDIOCORE_ALLOW_EMPTY=1 — the default config lists no models, so
#     the server starts healthy but cannot serve requests. Useful for
#     smoke-testing the image.
set -euo pipefail

CONFIG="${AUDIOCORE_CONFIG:-/etc/audiocore/server.json}"
DEFAULT="/etc/audiocore/server.json.default"

if [[ ! -f "${CONFIG}" ]]; then
    if [[ "${AUDIOCORE_ALLOW_EMPTY:-0}" == "1" && -f "${DEFAULT}" ]]; then
        echo "entrypoint: ${CONFIG} not found — copying default (no models)." >&2
        cp "${DEFAULT}" "${CONFIG}"
    else
        echo "entrypoint: config not found at ${CONFIG}." >&2
        echo "  Mount your server.json there, or set AUDIOCORE_CONFIG," >&2
        echo "  or set AUDIOCORE_ALLOW_EMPTY=1 for a no-model smoke test." >&2
        exit 2
    fi
fi

# JSON sanity check (python3 is not guaranteed in the runtime image; use jq
# if present, else fall through to the server's own parser).
if command -v jq >/dev/null 2>&1; then
    if ! jq empty "${CONFIG}" 2>/dev/null; then
        echo "entrypoint: ${CONFIG} is not valid JSON." >&2
        exit 2
    fi
fi

# Warn (don't fail) on any model path that is missing — the server itself
# will fail with a clearer per-model error, but a heads-up at startup helps.
if command -v jq >/dev/null 2>&1; then
    while IFS= read -r p; do
        [[ -z "${p}" ]] && continue
        if [[ ! -e "${p}" ]]; then
            echo "entrypoint: WARNING — model path does not exist: ${p}" >&2
        fi
    done < <(jq -r '.models[].path // empty' "${CONFIG}" 2>/dev/null)
fi

exec "$@"
