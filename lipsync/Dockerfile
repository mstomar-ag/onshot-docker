# =============================================================================
# OnShot Lipsync Pod — pre-baked Docker image (v1.6.6)
#
# This image is what RunPod pulls when launching a lipsync pod. Everything
# deploy_latentsync_comfyui.sh did live (clone, pip install, checkpoint
# downloads, three patches) is baked at build time. Pod boot becomes "pull
# image (cached after first pull) + start two servers" — ~2 min instead of
# the 30+ min flaky setup that bit cycle-wala / bandhani-jaipur /
# chai-aur-chemistry runs.
#
# DESIGN PRINCIPLE (v1.6.6)
# -------------------------
# Mirror deploy_latentsync_comfyui.sh EXACTLY. That script worked yesterday
# at ~5 min/shot. v1.6.2 → v1.6.5 deviated from it (different base, pinned
# torch + numpy, etc.) and either ran 4× slower or crash-looped. The script
# is the contract; the Dockerfile is the contract baked at build time.
#
# Specifically:
#   * BASE: runpod/comfyui:latest  (NOT runpod/pytorch — runpod/comfyui
#     ships ComfyUI + the right torch/cv2/numpy stack pre-installed).
#   * pip: NO version pins anywhere except `mediapipe<0.10.15` (the only
#     pin the script applies). Let pip resolve compatible versions on top
#     of the base, exactly as the script does.
#   * --break-system-packages: required on Ubuntu 24+ PEP 668 protection.
#   * Find ComfyUI dir at build time via the same fallback chain the
#     script uses (/workspace/ComfyUI → /workspace/runpod-slim/ComfyUI →
#     /comfyui), then symlink /workspace/ComfyUI for downstream paths.
#
# WHAT THIS IMAGE CONTAINS
#   /workspace/ComfyUI/                              — symlink to actual dir
#   .../custom_nodes/ComfyUI-LatentSyncWrapper       — patched (CI-O-22 + no-face)
#   .../custom_nodes/ComfyUI-VideoHelperSuite        — required for VHS_LoadVideo
#   .../checkpoints/latentsync_unet.pt               — 5 GB
#   .../checkpoints/stable_syncnet.pt                — 1.6 GB
#   .../checkpoints/whisper/tiny.pt                  — ~75 MB
#   .../checkpoints/vae/{config.json,safetensors}    — 335 MB
#   /workspace/CodeFormer                            — repo + 3 face weights
#   /workspace/gfpgan_weights/GFPGANv1.4.pth         — 332 MB
#   /opt/api_server_postprocess.py                   — :8000 sidecar
#   /entrypoint.sh                                   — boots ComfyUI + sidecar
#
# WHAT THIS IMAGE INTENTIONALLY DOES NOT CONTAIN
#   * RealESRGAN_x4plus_anime_6B.pth — optional bg upsampler for CodeFormer.
#     Postprocess server gracefully degrades when missing.
#   * Any secrets — image is publishable to a public GHCR registry.
# =============================================================================

# -----------------------------------------------------------------------------
# Single stage on runpod/comfyui:latest. Multi-stage doesn't help when the
# base already has ComfyUI + torch + cv2 + numpy baked in — there's nothing
# to "build" separately, just packages to layer on top.
# -----------------------------------------------------------------------------
FROM runpod/comfyui:latest

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Base apt deps (minimal — base already has python, ffmpeg, libgl, etc.) ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# --- Resolve ComfyUI path ---
# The runpod/comfyui:latest image lazy-installs ComfyUI on container boot
# (not at image build), so the deploy script's runtime fallback chain
# (/workspace/ComfyUI → /workspace/runpod-slim/ComfyUI → /comfyui) finds
# nothing during `docker build`. Strategy:
#   1. Search the whole image for any ComfyUI directory (covers any path).
#   2. If absent, clone it ourselves at /workspace/ComfyUI — same effect as
#      the runtime install would have, except baked at build time so we
#      don't pay the install cost on every pod boot.
# Either way, downstream steps see a working tree at /workspace/ComfyUI.
ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_REF=v0.3.30
RUN COMFY_DIR=$(find / -maxdepth 5 -type d -name ComfyUI 2>/dev/null | grep -v 'custom_nodes/ComfyUI' | head -1) \
 && if [ -n "$COMFY_DIR" ]; then \
        echo "Found ComfyUI in base image at $COMFY_DIR"; \
    else \
        echo "ComfyUI not in base image; cloning $COMFYUI_REPO @ $COMFYUI_REF"; \
        git clone --depth 1 --branch "$COMFYUI_REF" "$COMFYUI_REPO" /workspace/ComfyUI; \
        pip install --break-system-packages -r /workspace/ComfyUI/requirements.txt; \
        COMFY_DIR=/workspace/ComfyUI; \
    fi \
 && if [ "$COMFY_DIR" != "/workspace/ComfyUI" ]; then \
        rm -rf /workspace/ComfyUI 2>/dev/null || true; \
        mkdir -p /workspace; \
        ln -s "$COMFY_DIR" /workspace/ComfyUI; \
    fi \
 && ls -ld /workspace/ComfyUI && ls /workspace/ComfyUI/custom_nodes 2>/dev/null | head -5 || mkdir -p /workspace/ComfyUI/custom_nodes

# --- Step 1: ComfyUI-LatentSyncWrapper + its deps (mirrors deploy script:130-170) ---
# IMPORTANT: NO version pins. The deploy script runs `pip install -r
# requirements.txt`, then `pip install --upgrade torchcodec soundfile`, then
# `pip install insightface onnxruntime-gpu accelerate`. We do the same.
ARG LATENTSYNC_WRAPPER_REPO=https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
RUN cd /workspace/ComfyUI/custom_nodes \
 && git clone --depth 1 "$LATENTSYNC_WRAPPER_REPO" ComfyUI-LatentSyncWrapper \
 && cd ComfyUI-LatentSyncWrapper \
 && pip install --break-system-packages -r requirements.txt \
 && pip install --break-system-packages --upgrade soundfile \
 && pip install --break-system-packages insightface onnxruntime-gpu accelerate \
 && pip install --break-system-packages librosa

# --- FIX (v1.6.11-optfix3): match torchcodec to the cu128 torch stack ---
# The base ships a complete CUDA-12.8 torch stack (libnvrtc.so.12 present). The previous
# `--upgrade torchcodec` pulled the LATEST torchcodec, which is a CUDA-13 build needing
# libnvrtc.so.13 (absent) → ComfyUI's LoadAudio node crashes. Install the torchcodec built
# for cu128 from PyTorch's matching index so it uses the present libnvrtc.so.12. --no-deps
# keeps the working torch 2.11+cu128 untouched. (Can't import-verify on the CPU build runner —
# validated at pod runtime.)
RUN pip install --break-system-packages --force-reinstall --no-deps \
        torchcodec --index-url https://download.pytorch.org/whl/cu128 \
 && echo "torchcodec (cu128) installed: $(python3 -c 'import importlib.metadata as m; print(m.version(\"torchcodec\"))' 2>/dev/null || echo '?')"

# --- Step 2: ComfyUI-VideoHelperSuite (mirrors deploy script:240-256) ---
ARG VIDEO_HELPER_REPO=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
RUN cd /workspace/ComfyUI/custom_nodes \
 && git clone --depth 1 "$VIDEO_HELPER_REPO" ComfyUI-VideoHelperSuite \
 && cd ComfyUI-VideoHelperSuite \
 && pip install --break-system-packages -r requirements.txt

# --- Step 2b/2c/2d: Apply LatentSync patches (no-face, CI-O-22, CI-O-22c) ---
COPY patches/ /tmp/patches/
RUN python3 /tmp/patches/patch_ci_o_22_write_video.py \
        /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper \
 && python3 /tmp/patches/patch_no_face_graceful.py \
        /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper \
 && python3 /tmp/patches/patch_latentsync_read_video.py \
        /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper

# Build-time verification — fail LOUDLY if any patch silently no-op'd.
RUN grep -q "CI-O-22 PATCH APPLIED" /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py \
        || (echo "FATAL: CI-O-22 patch did not apply" && exit 1) \
 && grep -q "CI-O-22c" /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py \
        || (echo "FATAL: CI-O-22c marker missing" && exit 1) \
 && grep -q "no_face_indices" /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync/pipelines/lipsync_pipeline.py \
        || (echo "FATAL: no-face patch did not apply to lipsync_pipeline.py" && exit 1) \
 && grep -q "return None, None, None" /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync/utils/image_processor.py \
        || (echo "FATAL: no-face patch A did not apply to image_processor.py" && exit 1)

# --- Step 5: LatentSync core checkpoints (mirrors deploy script:540-571) ---
# Each download enforces a minimum size — fail the build than ship a corrupt image.
ENV CKPT=/workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints
RUN mkdir -p "$CKPT/whisper" "$CKPT/vae"
RUN curl -fL --retry 5 --retry-delay 10 --max-time 1800 \
        -o "$CKPT/latentsync_unet.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt' \
 && [ "$(stat -c%s $CKPT/latentsync_unet.pt)" -gt 5000000000 ] \
        || (echo "FATAL: latentsync_unet.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o "$CKPT/stable_syncnet.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt' \
 && [ "$(stat -c%s $CKPT/stable_syncnet.pt)" -gt 1500000000 ] \
        || (echo "FATAL: stable_syncnet.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 120 \
        -o "$CKPT/whisper/tiny.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt' \
 && [ "$(stat -c%s $CKPT/whisper/tiny.pt)" -gt 70000000 ] \
        || (echo "FATAL: whisper/tiny.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 60 \
        -o "$CKPT/latentsync_config.json" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/config.json'

# --- Step 1b: VAE in directory format (mirrors deploy script:172-238) ---
RUN curl -fL --retry 5 --retry-delay 10 --max-time 60 \
        -o "$CKPT/vae/config.json" \
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json' \
 && curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o "$CKPT/vae/diffusion_pytorch_model.safetensors" \
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors' \
 && [ "$(stat -c%s $CKPT/vae/diffusion_pytorch_model.safetensors)" -gt 300000000 ] \
        || (echo "FATAL: VAE safetensors too small" && exit 1)

# --- Step 6: Post-processing tools (mirrors deploy script:573-625) ---
# IMPORTANT: deploy script uses `pip install --break-system-packages basicsr
# facexlib gfpgan lpips realesrgan 'mediapipe<0.10.15'`. NO other version pins.
RUN pip install --break-system-packages \
        basicsr facexlib gfpgan lpips realesrgan 'mediapipe<0.10.15' \
        fastapi uvicorn python-multipart

# basicsr torchvision compat sed-fix (mirrors deploy script:590-594).
# Find the file via `find /usr -path '*/basicsr/data/degradations.py'` since
# basicsr might land in /usr/lib or /usr/local/lib depending on pip's choice.
RUN DEG=$(find /usr -path '*/basicsr/data/degradations.py' 2>/dev/null | head -1) \
 && if [ -n "$DEG" ] && grep -q 'from torchvision.transforms.functional_tensor import rgb_to_grayscale' "$DEG" 2>/dev/null; then \
        sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' "$DEG"; \
        echo "  Patched basicsr torchvision compat at $DEG"; \
    else \
        echo "  basicsr already patched or import path differs (DEG=$DEG)"; \
    fi \
 && python3 -c "import gfpgan; import basicsr.data.degradations; import mediapipe; print('postproc deps OK')"

# --- CodeFormer + facelib weights (mirrors deploy script:600-610) ---
RUN git clone --depth 1 https://github.com/sczhou/CodeFormer.git /workspace/CodeFormer \
 && printf '__version__ = "0.1.0"\n__gitsha__ = "unknown"\n' > /workspace/CodeFormer/basicsr/version.py \
 && mkdir -p /workspace/CodeFormer/weights/CodeFormer /workspace/CodeFormer/weights/facelib \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/CodeFormer/codeformer.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth' \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/facelib/detection_Resnet50_Final.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth' \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/facelib/parsing_parsenet.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'

# --- GFPGAN weights (mirrors deploy script:613-615) ---
RUN mkdir -p /workspace/gfpgan_weights \
 && curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o /workspace/gfpgan_weights/GFPGANv1.4.pth \
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

# --- FIX (v1.6.11-optfix): un-shadow the baked tree ---------------------------
# RunPod mounts the pod volume over /workspace at boot, which SHADOWS everything
# this image baked there (ComfyUI + custom_nodes + 7 GB checkpoints + CodeFormer +
# GFPGAN weights) — so LatentSyncNode never registers and the postprocess weights
# vanish. /opt is NOT a mount point, so we stash the whole baked /workspace there;
# the entrypoint restores it into the (mounted, empty) /workspace before starting
# the servers. `mv` (not `cp`) keeps the image the same size — the data lives once.
# See onshot-lipsync docs/runpod-image-setup.md for the full root cause.
RUN mv /workspace /opt/baked_workspace && mkdir -p /workspace \
 && echo "stashed baked tree to /opt/baked_workspace ($(du -sh /opt/baked_workspace | cut -f1))"

# --- Postprocess server + entrypoint ---
COPY api_server_postprocess.py /opt/api_server_postprocess.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# CI-O-21 safety net: torchaudio.save() uses soundfile backend
ENV TORCHAUDIO_USE_BACKEND_DISPATCHER=1 \
    TORCHAUDIO_BACKEND=soundfile \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="onshot-lipsync-latentsync"
LABEL org.opencontainers.image.description="OnShot lipsync pod — ComfyUI + LatentSync 1.6 + post-processing sidecar, fully baked"
LABEL org.opencontainers.image.source="https://github.com/Agrim-Intelligence/onshot-docker"
LABEL org.opencontainers.image.licenses="proprietary"
LABEL org.opencontainers.image.version="1.6.11-optfix3"
LABEL onshot.role="lipsync"
LABEL onshot.engine="latentsync-1.6"

# Ports: 8188 = ComfyUI, 8000 = postprocess sidecar
EXPOSE 8188 8000
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
