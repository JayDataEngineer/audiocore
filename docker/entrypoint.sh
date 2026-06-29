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
#   - /etc/audiocore/server.json.default (shipped copy) — seeded automatically
#     because the default config sets "model_dir": "/models" and auto-discovers
#     whatever the user mounts there. Boots idle if /models is empty.
#
# AUDIOCORE_ALLOW_EMPTY is tolerated for backwards compat but no longer
# required — the default config is always safe to seed.
set -euo pipefail

CONFIG="${AUDIOCORE_CONFIG:-/etc/audiocore/server.json}"
DEFAULT="/etc/audiocore/server.json.default"

if [[ ! -f "${CONFIG}" ]]; then
    # The shipped default config sets "model_dir": "/models" and "models": [],
    # which means the server auto-discovers whatever is under /models at boot
    # and boots idle if nothing is there. Both states are safe, so we always
    # seed the default when the user didn't mount their own config — Docker
    # UX is just "mount /models and run".
    if [[ -f "${DEFAULT}" ]]; then
        echo "entrypoint: ${CONFIG} not found — seeding from default (auto-discovers /models)." >&2
        cp "${DEFAULT}" "${CONFIG}"
    else
        echo "entrypoint: config not found at ${CONFIG} and no default shipped." >&2
        echo "  Mount your server.json at ${CONFIG} or set AUDIOCORE_CONFIG." >&2
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

# Inject --config $CONFIG when launching the canonical server without one,
# so "docker run audiocore" Just Works. If the user passed --config on the
# command line, or is invoking a different binary (cli, inspect_gguf, sh),
# we get out of the way (postgres/redis-style wrapper).
case "${1:-}" in
    audiocore_server|/opt/audiocore/bin/audiocore_server)
        has_config=0
        for a in "$@"; do
            case "$a" in --config|--config=*) has_config=1 ;; esac
        done
        [[ $has_config -eq 0 ]] && set -- "$@" --config "${CONFIG}"
        ;;
esac

exec "$@"
