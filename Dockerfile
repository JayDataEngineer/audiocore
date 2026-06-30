# syntax=docker/dockerfile:1.7
#
# audiocore — multi-stage CUDA build.
#
#   docker build -t audiocore .
#   docker run --gpus all -p 8080:8080 \
#     -v /path/to/models:/models:ro \
#     -v /path/to/server.json:/etc/audiocore/server.json:ro \
#     audiocore
#
# Build knobs (override with --build-arg):
#   CUDA_VERSION   13.1.0          base CUDA tag (must match your driver)
#   UBUNTU         24.04           base Ubuntu tag
#   ENABLE_MP3     ON              libmp3lame MP3 output
#   BUILD_TYPE     Release         RelWithDebInfo | Debug | Release
#   BUILD_JOBS     (auto)          cores used for cmake --build
#
# For a CPU-only image (no CUDA, ~350 MB), use Dockerfile.cpu.

ARG CUDA_VERSION=13.1.0
ARG UBUNTU=24.04

# ─── Builder ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        git \
        pkg-config \
        libmp3lame-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /src

# Copy the source tree. third_party/llama.cpp and third_party/ggml MUST be
# populated first (git submodule update --init --recursive) — they are git
# submodules and are not stored as files in this repo.
COPY . .

ARG ENABLE_MP3=ON
ARG BUILD_TYPE=Release
ARG BUILD_JOBS=

RUN : "${BUILD_JOBS:=$(nproc)}" \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DENGINE_ENABLE_CUDA=ON \
        -DENGINE_ENABLE_MP3=${ENABLE_MP3} \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_CLI=ON \
        -DENGINE_BUILD_SERVER=ON \
    && cmake --build build -j"${BUILD_JOBS}" \
        --target audiocore_server audiocore_cli convert_acestep convert_qwen3tts inspect_gguf

# Strip the binaries in-place (Release already, but kill the rest of the
# symbol/section table for a smaller runtime image).
RUN find build -maxdepth 1 -type f -perm -u+x -exec strip --strip-unneeded {} \; 2>/dev/null || true

# ─── Runtime ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmp3lame0 \
        libgomp1 \
        ca-certificates \
        curl \
        jq \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10000 audiocore \
    && useradd  --system --uid 10000 --gid 10000 \
                --home-dir /var/lib/audiocore --shell /usr/sbin/nologin audiocore

# Binaries (built at the build/ root; llama.cpp's CMakeLists only scopes
# CMAKE_RUNTIME_OUTPUT_DIRECTORY to its own subdirectory).
COPY --from=builder /src/build/audiocore_server  /opt/audiocore/bin/audiocore_server
COPY --from=builder /src/build/audiocore_cli     /opt/audiocore/bin/audiocore_cli
COPY --from=builder /src/build/convert_acestep   /opt/audiocore/bin/convert_acestep
COPY --from=builder /src/build/convert_qwen3tts  /opt/audiocore/bin/convert_qwen3tts
COPY --from=builder /src/build/inspect_gguf      /opt/audiocore/bin/inspect_gguf

# Shared libs from the ggml/llama.cpp build. COPY preserves the SONAME
# symlinks (libggml.so -> libggml.so.0 -> libggml.so.0.15.3) when they
# point within the same source directory.
COPY --from=builder /src/build/bin/lib*.so* /opt/audiocore/lib/

# Entrypoint + container-friendly default config.
COPY docker/entrypoint.sh        /opt/audiocore/bin/entrypoint.sh
COPY docker/server.example.json  /etc/audiocore/server.json.default
# Model-fetch utilities (opt-in via AUDIOCORE_PREPULL=1). fetch_models.sh is
# pure bash + curl + jq + git; manifest records repo / sha256 / converter.
COPY scripts/fetch_models.sh     /opt/audiocore/scripts/fetch_models.sh
COPY models/manifest.json        /opt/audiocore/models/manifest.json
RUN  chmod +x /opt/audiocore/bin/entrypoint.sh /opt/audiocore/scripts/fetch_models.sh \
    && mkdir -p /etc/audiocore /models /var/lib/audiocore \
    && chown -R audiocore:audiocore /opt/audiocore /etc/audiocore /models /var/lib/audiocore

ENV LD_LIBRARY_PATH=/opt/audiocore/lib \
    PATH=/opt/audiocore/bin:${PATH} \
    AUDIOCORE_HOST=0.0.0.0 \
    AUDIOCORE_PORT=8080 \
    AUDIOCORE_CONFIG=/etc/audiocore/server.json \
    AUDIOCORE_BUILD_DIR=/opt/audiocore/bin

USER audiocore
WORKDIR /var/lib/audiocore

EXPOSE 8080

# /health returns 200 once the model registry has loaded. start-period is
# generous because loading GGUFs into VRAM takes time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${AUDIOCORE_PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/audiocore/bin/entrypoint.sh"]
CMD ["audiocore_server"]
