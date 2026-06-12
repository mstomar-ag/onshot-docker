#!/bin/bash
# OnShot lipsync pod entrypoint.
#
# Boots two cooperating servers in the same container:
#   :8188 — ComfyUI (LatentSync inference engine)
#   :8000 — api_server_postprocess.py sidecar (GFPGAN/CodeFormer/feathered-blend)
#
# Both write to /workspace/*.log; the trailing `tail -F` pipes logs into the
# container's stdout so `docker logs` (and RunPod's web UI) sees them.
#
# Boot resilience (CI-O-58w): if either server exits, we DO NOT exit 1 — we
# `sleep infinity` so the container stays up and SSH/proxy traffic keeps
# working, allowing live debugging of the failure (logs in /workspace, env
# inspection, etc). Without this, a runtime import error (e.g. cv2/numpy ABI
# mismatch) crash-loops the container forever — no SSH, no port proxy,
# nothing to grab logs from.
# NB: NO `set -e` — a failed `cd` to a missing dir would otherwise exit the
# container before sleep-infinity, which is exactly the v1.6.6 crash-loop bug:
# entrypoint hard-coded `cd /opt/ComfyUI` but the build cloned ComfyUI to
# /workspace/ComfyUI when find didn't locate it in the runpod/comfyui base.
# We resolve the path dynamically below and tolerate failures so the
# diagnostic preflight + sleep-infinity tail always run.

mkdir -p /workspace
cd /workspace

# --- FIX (v1.6.11-optfix): restore the baked tree shadowed by RunPod's /workspace mount ---
# RunPod mounts the pod volume over /workspace at boot, hiding everything baked there at
# build time. We stashed it to /opt/baked_workspace (not a mount point) in the Dockerfile;
# restore it now, BEFORE resolving ComfyUI / starting servers. The guard makes this a no-op
# on a persistent volume that's already populated (only restores when /workspace is empty).
if [ -d /opt/baked_workspace ] && [ ! -e /workspace/ComfyUI ]; then
    echo "[entrypoint] /workspace empty (shadowed by RunPod mount) — restoring baked tree from /opt/baked_workspace ..."
    cp -a /opt/baked_workspace/. /workspace/ \
        && echo "[entrypoint] restore complete ($(du -sh /workspace 2>/dev/null | cut -f1))" \
        || echo "[entrypoint] WARNING: restore failed"
fi

# CI-O-21 safety net (also baked into Dockerfile ENV, kept here for clarity).
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1
export TORCHAUDIO_BACKEND=soundfile

# --- Start sshd if available (runpod/comfyui base ships sshd but its own
# entrypoint starts it — we overrode that, so do it ourselves). Inject the
# RunPod-supplied PUBLIC_KEY into root's authorized_keys so the operator can
# SSH in for log inspection. Without this, all the "sleep infinity" debug
# affordance below is unreachable.
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
if command -v sshd >/dev/null 2>&1; then
    mkdir -p /run/sshd
    # Generate host keys if missing (fresh container).
    [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd -D > /workspace/sshd.log 2>&1 &
    echo "[entrypoint] sshd started (pid=$!)"
fi

# Resolve ComfyUI dir at boot.
#
# v1.6.9 FIX (kolkata-diwali-alpana-arjun, 2026-05-11):
# The runpod/comfyui base image places ComfyUI at /opt/comfyui-baked (note
# the lowercase + `-baked` suffix) and our Dockerfile creates a build-time
# symlink at /workspace/ComfyUI → /opt/comfyui-baked. But RunPod mounts
# /workspace as a network filesystem on pod boot, which **shadows** the
# build-time symlink — so the symlink is invisible at runtime.
#
# Prior versions only searched paths under /workspace and /opt/ComfyUI
# (capital C), missed /opt/comfyui-baked, fell back to a case-sensitive
# */ComfyUI/main.py find, missed it again, then dropped into degraded
# sleep-only mode with no ComfyUI. Symptom: every v1.6.8 pod had :8000
# sidecar healthy but :8188 ComfyUI 502 forever.
#
# Fix: search /opt/comfyui-baked FIRST (it's where the base image puts
# ComfyUI), then the other candidate paths, and use a broader case-
# insensitive find as the last-resort fallback.
COMFY_DIR=""
# v1.6.11-optfix: PREFER a ComfyUI that actually has the LatentSync custom node — after the
# restore that's /workspace/ComfyUI. This avoids picking an empty base-image ComfyUI (e.g.
# /opt/comfyui-baked) that has main.py but no custom_nodes (the original silent-fail mode).
for d in /workspace/ComfyUI /opt/comfyui-baked /workspace/runpod-slim/ComfyUI /opt/ComfyUI /comfyui; do
    if [ -f "$d/main.py" ] && [ -d "$d/custom_nodes/ComfyUI-LatentSyncWrapper" ]; then
        COMFY_DIR="$d"; break
    fi
done
# Fallback 1: any ComfyUI with main.py (degraded — LatentSyncNode may be missing).
if [ -z "$COMFY_DIR" ]; then
    for d in /workspace/ComfyUI /opt/comfyui-baked /workspace/runpod-slim/ComfyUI /opt/ComfyUI /comfyui; do
        if [ -f "$d/main.py" ]; then COMFY_DIR="$d"; break; fi
    done
fi
# Fallback 2: broad find (covers nested base-image layouts + case variants).
if [ -z "$COMFY_DIR" ]; then
    COMFY_DIR=$(find / -maxdepth 6 -type f -name main.py 2>/dev/null \
        | grep -iE '/(comfyui|comfyui-baked|comfy)/main\.py$' \
        | head -1)
    [ -n "$COMFY_DIR" ] && COMFY_DIR=$(dirname "$COMFY_DIR")
fi
echo "[entrypoint] ComfyUI dir resolved: ${COMFY_DIR:-<NOT FOUND>}"

# Re-establish the /workspace/ComfyUI convenience symlink for downstream
# consumers (paths that hard-code /workspace/ComfyUI/custom_nodes/...).
# Build-time we created this symlink, but RunPod's network-volume mount of
# /workspace at boot hides it — recreate it now if missing.
if [ -n "$COMFY_DIR" ] && [ "$COMFY_DIR" != "/workspace/ComfyUI" ] && [ ! -e "/workspace/ComfyUI" ]; then
    ln -sfn "$COMFY_DIR" /workspace/ComfyUI
    echo "[entrypoint] recreated /workspace/ComfyUI -> $COMFY_DIR symlink (network-volume shadowed build-time symlink)"
fi

# --- Pre-flight import checks (CI-O-58w) ---
# Validate the runtime stack BEFORE we kick off the heavy services. Any
# ImportError here lands in /workspace/preflight.log with a clean traceback,
# instead of being buried in ComfyUI's startup spam after a crash.
{
  echo "=== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 - <<'PYEOF' 2>&1
import sys; print("python:", sys.version)
import importlib
mods = [
    "torch", "torchvision", "torchaudio", "torchcodec",
    "numpy", "scipy", "cv2", "PIL",
    "diffusers", "transformers", "accelerate", "safetensors",
    "mediapipe", "librosa", "imageio", "soundfile",
    "basicsr", "facexlib", "gfpgan", "lpips", "realesrgan",
    "fastapi", "uvicorn", "onnxruntime",
]
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {name}=={v}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
PYEOF
} > /workspace/preflight.log 2>&1
echo "[entrypoint] preflight done; see /workspace/preflight.log"

# --- Start ComfyUI on :8188 ---
if [ -n "$COMFY_DIR" ] && [ -f "$COMFY_DIR/main.py" ]; then
    cd "$COMFY_DIR"
    python3 main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header \
        --disable-auto-launch \
        > /workspace/comfyui.log 2>&1 &
    COMFY_PID=$!
else
    echo "[entrypoint] FATAL: ComfyUI main.py not found anywhere — starting in degraded mode (sleep only). preflight.log will show what the image contains." \
        > /workspace/comfyui.log
    COMFY_PID=""
fi

# --- Start postprocess sidecar on :8000 ---
nohup python3 /opt/api_server_postprocess.py \
    > /workspace/postprocess.log 2>&1 &
POSTPROC_PID=$!

echo "[entrypoint] ComfyUI pid=${COMFY_PID:-<none>}, Postprocess pid=${POSTPROC_PID:-<none>}"

# Stream all three logs to container stdout so `docker logs` shows them live.
tail -F /workspace/comfyui.log /workspace/postprocess.log /workspace/preflight.log 2>/dev/null &
TAIL_PID=$!

# Wait on either real server dying (ignore the tail — it lives forever).
# `wait -n` returns when the first job exits; we DO NOT propagate that as a
# container exit, so SSH stays up for debugging.
PIDS=""
[ -n "$COMFY_PID" ] && PIDS="$PIDS $COMFY_PID"
[ -n "$POSTPROC_PID" ] && PIDS="$PIDS $POSTPROC_PID"
if [ -n "$PIDS" ]; then
    wait -n $PIDS
    EXIT_CODE=$?
else
    EXIT_CODE=255
    echo "[entrypoint] no servers started — degraded mode"
fi

echo "[entrypoint] one of the servers exited (code=$EXIT_CODE); container will sleep" \
     "indefinitely so SSH + log inspection stay available. Inspect" \
     "/workspace/comfyui.log /workspace/postprocess.log /workspace/preflight.log."
kill -TERM $TAIL_PID 2>/dev/null || true

# Keep PID 1 alive forever so the container doesn't crash-loop. Operator can
# `docker logs <id>` or SSH in to see what went wrong.
sleep infinity
