"""
Lightweight FastAPI post-processing server for LatentSync ComfyUI.

Runs alongside ComfyUI on the same pod (port 8000). Accepts raw lip-synced
video + original video, applies requested post-processing pipeline, returns
the final result.

Post-processing order:
  1. GFPGAN face restoration (pre-blend)
  2. Feathered blend (re-blend mouth into original background)
  3. CodeFormer face restoration (post-blend)
  4. Temporal smoothing (anti-jitter)

Extracted from api_server_latentsync_v1.py — same algorithms, no inference.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:
    raise SystemExit("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))
# Model paths match deploy_latentsync_comfyui.sh layout
GFPGAN_MODEL_PATH = WORKSPACE / "gfpgan_weights" / "GFPGANv1.4.pth"
CODEFORMER_MODELS = WORKSPACE / "CodeFormer" / "weights" / "facelib"  # detection + parsing models
CODEFORMER_WEIGHTS = WORKSPACE / "CodeFormer" / "weights" / "CodeFormer"  # codeformer.pth
TEMP_DIR = WORKSPACE / "postprocess_tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("postprocess")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

app = FastAPI(title="LipSync Post-Processing Server", version="1.0.0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mp_face_mesh():
    """Return mp.solutions.face_mesh module, or None if unavailable."""
    try:
        import mediapipe as mp
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            return None
        return getattr(solutions, "face_mesh", None)
    except ImportError:
        return None


_realesrgan_upsampler_cache = None
_realesrgan_lock: Optional[threading.Lock] = None


def _get_realesrgan_upsampler():
    """Lazy singleton: load RealESRGAN x4 anime model once, reuse."""
    global _realesrgan_upsampler_cache, _realesrgan_lock
    if _realesrgan_lock is None:
        _realesrgan_lock = threading.Lock()
    with _realesrgan_lock:
        if _realesrgan_upsampler_cache is not None:
            return _realesrgan_upsampler_cache
        model_path = Path("/workspace/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth")
        if not model_path.exists():
            logger.warning("RealESRGAN model not found — CodeFormer won't use bg upsampler")
            return None
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=str(model_path), model=model,
                tile=256, tile_pad=10, pre_pad=0, half=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            _realesrgan_upsampler_cache = upsampler
            logger.info("RealESRGAN x4 anime loaded")
            return upsampler
        except Exception as e:
            logger.warning(f"RealESRGAN load failed: {e}")
            return None


def _ffmpeg_merge_audio(video_no_audio: str, audio_source: str, output: str):
    """Merge video (no audio) with audio from source using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", video_no_audio,
         "-i", audio_source,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac",
         "-map", "0:v", "-map", "1:a?",
         "-shortest",
         output],
        check=True, capture_output=True, timeout=300,
    )


# ---------------------------------------------------------------------------
# Post-processing functions (extracted from api_server_latentsync_v1.py)
# ---------------------------------------------------------------------------

def apply_gfpgan(input_video: str, output_video: str) -> bool:
    """GFPGAN face restoration. Returns True on success."""
    if not GFPGAN_MODEL_PATH.exists():
        logger.warning(f"GFPGAN model not found at {GFPGAN_MODEL_PATH}")
        return False

    try:
        # Suppress basicsr ARCH_REGISTRY conflicts when gfpgan re-imports archs.
        # Both gfpgan and CodeFormer register overlapping arch names (ResNetArcFace, etc.).
        # Monkey-patch the registry to allow overwrites during import.
        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            _orig_register = ARCH_REGISTRY.register
            def _safe_register(cls=None, **kwargs):
                try:
                    return _orig_register(cls, **kwargs)
                except KeyError:
                    # Already registered — return the class as-is (decorator pattern)
                    return cls if cls is not None else (lambda c: c)
            ARCH_REGISTRY.register = _safe_register
        except Exception:
            pass

        from gfpgan import GFPGANer

        # Restore original register
        try:
            ARCH_REGISTRY.register = _orig_register
        except Exception:
            pass
    except ImportError:
        logger.warning("gfpgan not installed")
        return False

    restorer = GFPGANer(
        model_path=str(GFPGAN_MODEL_PATH),
        upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
    )

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp = str(Path(output_video).with_suffix(".gfpgan.mp4"))
    writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    count = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            _, _, enhanced = restorer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            writer.write(enhanced)
            count += 1
        except Exception:
            writer.write(frame)
        idx += 1
        if idx % 50 == 0:
            logger.info(f"  GFPGAN: {idx}/{total}")

    cap.release()
    writer.release()

    if count == 0:
        Path(temp).unlink(missing_ok=True)
        return False

    _ffmpeg_merge_audio(temp, input_video, output_video)
    Path(temp).unlink(missing_ok=True)
    logger.info(f"  GFPGAN: {count}/{total} frames enhanced")
    return True


def _get_face_analyzer():
    """Lazy-load InsightFace buffalo_l for per-frame face pose (yaw/pitch/roll).

    buffalo_l includes landmark_3d_68 which provides head pose estimation.
    buffalo_sc only has detection — no pose.
    """
    cache_key = "_face_analyzer"
    if not hasattr(_get_face_analyzer, cache_key):
        try:
            from insightface.app import FaceAnalysis
            fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            fa.prepare(ctx_id=0, det_size=(640, 640))
            setattr(_get_face_analyzer, cache_key, fa)
            logger.info("InsightFace buffalo_l loaded (with pose estimation)")
        except Exception as e:
            logger.warning(f"InsightFace buffalo_l unavailable ({e}) — trying buffalo_sc fallback")
            try:
                from insightface.app import FaceAnalysis
                fa = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                fa.prepare(ctx_id=0, det_size=(320, 320))
                setattr(_get_face_analyzer, cache_key, fa)
                logger.info("InsightFace buffalo_sc loaded (no pose — confidence-only fallback)")
            except Exception as e2:
                logger.warning(f"InsightFace unavailable ({e2}) — crossfade disabled")
                setattr(_get_face_analyzer, cache_key, None)
    return getattr(_get_face_analyzer, cache_key)


def apply_feathered_blend(
    original_video: str, lipsync_video: str, output_video: str,
    feather_radius: int = 35,
    min_face_confidence: float = 0.6,
    full_blend_confidence: float = 0.85,
) -> bool:
    """Re-blend lip-synced face into original background using feathered mask.

    Per-frame face confidence crossfade:
    - confidence >= full_blend_confidence → full lip sync blend
    - min_face_confidence <= confidence < full_blend_confidence → gradual crossfade
    - confidence < min_face_confidence → use original frame entirely

    This prevents blur artifacts on extreme head turns/bends where LatentSync
    produces low-quality output but the feathered blend would still paste it.
    """
    mp_face_mesh = _get_mp_face_mesh()
    if mp_face_mesh is None:
        logger.warning("mediapipe unavailable — skipping feathered blend")
        return False

    # Load InsightFace for per-frame face pose (yaw) detection
    face_analyzer = _get_face_analyzer()

    cap_orig = cv2.VideoCapture(original_video)
    cap_sync = cv2.VideoCapture(lipsync_video)
    if not cap_orig.isOpened() or not cap_sync.isOpened():
        cap_orig.release()
        cap_sync.release()
        return False

    fps = cap_sync.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap_sync.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_sync.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_sync.get(cv2.CAP_PROP_FRAME_COUNT))

    if feather_radius % 2 == 0:
        feather_radius += 1

    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    temp = str(Path(output_video).with_suffix(".blended.mp4"))
    writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    blended = 0
    original_used = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.4,
    ) as mesh:
        idx = 0
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_sync, frame_sync = cap_sync.read()
            if not ret_sync:
                break
            if not ret_orig:
                writer.write(frame_sync)
                idx += 1
                continue

            if frame_orig.shape[:2] != (h, w):
                frame_orig = cv2.resize(frame_orig, (w, h))

            # ── Per-frame face yaw check ──────────────────────────────
            # Use head yaw angle to decide blend weight.
            # Side profiles (yaw > 40°) produce blurry LatentSync → use original.
            # Frontal (yaw < 25°) → full lip sync. Gradual crossfade in between.
            MAX_YAW_FOR_FULL_BLEND = 25.0   # Below this → full lip sync
            MAX_YAW_FOR_ANY_BLEND = 45.0    # Above this → original only
            confidence = 1.0
            if face_analyzer is not None:
                try:
                    det_faces = face_analyzer.get(frame_sync)
                    if det_faces:
                        face = det_faces[0]
                        abs_yaw = abs(face.pose[0]) if hasattr(face, 'pose') else 0.0
                        det_score = face.det_score
                        # Use yaw for blend decision (more accurate than det_score)
                        if abs_yaw >= MAX_YAW_FOR_ANY_BLEND or det_score < min_face_confidence:
                            confidence = 0.0
                        elif abs_yaw >= MAX_YAW_FOR_FULL_BLEND:
                            # Gradual: 25°→45° maps to 1.0→0.0
                            confidence = 1.0 - (abs_yaw - MAX_YAW_FOR_FULL_BLEND) / (MAX_YAW_FOR_ANY_BLEND - MAX_YAW_FOR_FULL_BLEND)
                        else:
                            confidence = 1.0
                    else:
                        confidence = 0.0
                except Exception:
                    confidence = 0.0

            # Low confidence / high yaw → use original frame (no blur)
            if confidence < 0.05:
                writer.write(frame_orig)
                original_used += 1
                idx += 1
                continue

            try:
                rgb = cv2.cvtColor(frame_sync, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0]
                    points = np.array([
                        [int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                        for i in FACE_OVAL
                    ], dtype=np.int32)
                    hull = cv2.convexHull(points)
                    mask = np.zeros((h, w), dtype=np.float32)
                    cv2.fillConvexPoly(mask, hull, 1.0)
                    mask = cv2.GaussianBlur(mask, (feather_radius, feather_radius), 0)
                    mask_3ch = mask[:, :, np.newaxis]

                    # Full lip sync blend
                    synced = (frame_sync.astype(np.float32) * mask_3ch +
                              frame_orig.astype(np.float32) * (1.0 - mask_3ch))

                    # Gradual crossfade based on yaw-derived confidence
                    if confidence < 1.0:
                        out = synced * confidence + frame_orig.astype(np.float32) * (1.0 - confidence)
                    else:
                        out = synced

                    writer.write(out.astype(np.uint8))
                    blended += 1
                else:
                    # MediaPipe found no face — use original (not blurry sync)
                    writer.write(frame_orig)
                    original_used += 1
            except Exception:
                writer.write(frame_orig)
                original_used += 1

            idx += 1
            if idx % 100 == 0:
                logger.info(f"  Feathered blend: {idx}/{total}")

    cap_orig.release()
    cap_sync.release()
    writer.release()

    logger.info(f"  Feathered blend: {blended}/{total} frames blended, "
                f"{original_used} used original (low confidence)")

    if blended == 0 and original_used == 0:
        Path(temp).unlink(missing_ok=True)
        return False

    try:
        _ffmpeg_merge_audio(temp, lipsync_video, output_video)
    except Exception as e:
        logger.error(f"  Feathered blend ffmpeg failed: {e}")
        Path(temp).unlink(missing_ok=True)
        return False

    Path(temp).unlink(missing_ok=True)
    return True


def apply_codeformer(
    input_video: str, output_video: str,
    weight: float = 0.5, mouth_only: bool = False,
) -> bool:
    """CodeFormer face restoration. Returns True on success."""
    codeformer_model = CODEFORMER_WEIGHTS / "codeformer.pth"
    if not codeformer_model.exists():
        logger.warning(f"CodeFormer model not found at {codeformer_model}")
        return False

    try:
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError as e:
        logger.warning(f"CodeFormer deps missing ({e})")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CodeFormer arch from repo fork
    import sys as _sys
    net = None
    CODEFORMER_REPO = WORKSPACE / "CodeFormer"
    if CODEFORMER_REPO.exists():
        _orig_path = _sys.path[:]
        try:
            _sys.path.insert(0, str(CODEFORMER_REPO))
            _ver = CODEFORMER_REPO / "basicsr" / "version.py"
            if not _ver.exists():
                _ver.write_text('__version__ = "1.0.0"\n__gitsha__ = "unknown"\n')

            _cache_key = "_codeformer_class_cache"
            if not hasattr(apply_codeformer, _cache_key):
                import runpy
                import builtins as _builtins
                _archs_dir = CODEFORMER_REPO / "basicsr" / "archs"
                _vq_ns = runpy.run_path(str(_archs_dir / "vqgan_arch.py"))
                _cf_src = (_archs_dir / "codeformer_arch.py").read_text()
                _cf_src = _cf_src.replace(
                    "from basicsr.archs.vqgan_arch import *",
                    "pass  # patched: vqgan names injected",
                )
                _cf_ns = dict(_vq_ns)
                _cf_code = compile(_cf_src, str(_archs_dir / "codeformer_arch.py"), "exec")
                _builtins.exec(_cf_code, _cf_ns)
                setattr(apply_codeformer, _cache_key, _cf_ns["CodeFormer"])

            _CodeFormerClass = getattr(apply_codeformer, _cache_key)
            net = _CodeFormerClass(
                dim_embd=512, codebook_size=1024, n_head=8,
                n_layers=9, connect_list=["32", "64", "128", "256"],
            ).to(device)
        except Exception as e:
            logger.warning(f"CodeFormer repo-fork import failed ({e})")
            net = None
        finally:
            _sys.path = _orig_path

    if net is None:
        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            arch_name = next(
                (k for k in ARCH_REGISTRY._obj_map if "codeformer" in k.lower()), None)
            if arch_name is None:
                return False
            net = ARCH_REGISTRY.get(arch_name)(
                dim_embd=512, codebook_size=1024, n_head=8,
                n_layers=9, connect_list=["32", "64", "128", "256"],
            ).to(device)
        except Exception:
            return False

    try:
        ckpt = torch.load(str(codeformer_model), map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)), strict=False)
        net.eval()
    except Exception:
        return False

    try:
        from basicsr.utils import img2tensor, tensor2img
    except ImportError:
        def img2tensor(imgs, bgr2rgb=True, float32=True):
            if not isinstance(imgs, list):
                imgs = [imgs]
            result = []
            for img in imgs:
                if bgr2rgb:
                    img = img[:, :, ::-1]
                t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
                if float32:
                    t = t.float()
                result.append(t)
            return result[0] if len(result) == 1 else result

        def tensor2img(tensor, rgb2bgr=True, out_type=None, min_max=(0, 1)):
            if out_type is None:
                out_type = np.uint8
            t = tensor.squeeze(0).float().clamp_(min_max[0], min_max[1])
            t = (t - min_max[0]) / (min_max[1] - min_max[0])
            img = t.detach().cpu().numpy().transpose(1, 2, 0)
            if rgb2bgr and img.shape[2] == 3:
                img = img[:, :, ::-1]
            return (img * 255.0).round().astype(out_type)

    bg_upsampler = _get_realesrgan_upsampler()
    upscale = 2 if bg_upsampler is not None else 1

    parse_model = CODEFORMER_MODELS / "parsing_parsenet.pth"  # /workspace/CodeFormer/weights/facelib/
    has_parse = parse_model.exists() or Path("/workspace/models/facexlib/parsing_parsenet.pth").exists()

    try:
        face_helper = FaceRestoreHelper(
            upscale_factor=upscale, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png",
            use_parse=has_parse, device=device,
            model_rootpath=str(CODEFORMER_MODELS),
        )
    except Exception as e:
        logger.error(f"FaceRestoreHelper init failed: {e}")
        return False

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp = str(Path(output_video).with_suffix(".cf.mp4"))
    writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid))

    LIPS_ALL_CF = sorted(set([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        95, 88, 178, 87, 14, 317, 402, 318, 324,
    ]))

    mp_face_mesh = _get_mp_face_mesh()
    use_mouth_mask = mouth_only and mp_face_mesh is not None
    mesh_ctx = None
    if use_mouth_mask:
        mesh_ctx = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.4, min_tracking_confidence=0.4,
        )
        mesh_ctx.__enter__()

    count = 0
    idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                face_helper.clean_all()
                face_helper.read_image(frame)
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()

                for cropped_face in face_helper.cropped_faces:
                    face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    face_t = (face_t - 0.5) / 0.5
                    face_t = face_t.unsqueeze(0).to(device)
                    try:
                        output = net(face_t, w=weight, adain=True)[0]
                        restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception:
                        restored = tensor2img(face_t, rgb2bgr=True, min_max=(-1, 1))
                    restored = restored.astype(np.uint8)
                    # facexlib >= 0.3.0 changed signature: add_restored_face(face)
                    # older versions: add_restored_face(face, input_face)
                    try:
                        face_helper.add_restored_face(restored, cropped_face)
                    except TypeError:
                        face_helper.add_restored_face(restored)

                face_helper.get_inverse_affine(None)
                if bg_upsampler is not None:
                    bg_img = bg_upsampler.enhance(frame, outscale=upscale)[0]
                    cf_full = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, face_upsampler=bg_upsampler)
                else:
                    cf_full = face_helper.paste_faces_to_input_image()

                if upscale > 1:
                    cf_full = cv2.resize(cf_full, (w_vid, h_vid), interpolation=cv2.INTER_LANCZOS4)

                if use_mouth_mask:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_mp = mesh_ctx.process(rgb)
                    if result_mp.multi_face_landmarks:
                        lm = result_mp.multi_face_landmarks[0]
                        face_h_px = int((lm.landmark[152].y - lm.landmark[10].y) * h_vid)
                        lip_pts = []
                        for li in LIPS_ALL_CF:
                            if li < len(lm.landmark):
                                lk = lm.landmark[li]
                                lip_pts.append([int(lk.x * w_vid), int(lk.y * h_vid)])
                        if len(lip_pts) >= 6:
                            pts = np.array(lip_pts, dtype=np.int32)
                            hull = cv2.convexHull(pts)
                            raw_mask = np.zeros((h_vid, w_vid), dtype=np.uint8)
                            cv2.fillConvexPoly(raw_mask, hull, 255)
                            exp_px = max(6, int(face_h_px * 0.10))
                            k = exp_px * 2 + 1
                            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                            expanded = cv2.dilate(raw_mask, kern, iterations=1)
                            blur_r = max(5, int(face_h_px * 0.05))
                            if blur_r % 2 == 0:
                                blur_r += 1
                            mask_f = cv2.GaussianBlur(
                                expanded.astype(np.float32) / 255.0,
                                (blur_r, blur_r), 0,
                            )[:, :, np.newaxis]
                            mask_bool = mask_f[:, :, 0] > 0.01
                            if mask_bool.any():
                                lab_cf = cv2.cvtColor(cf_full, cv2.COLOR_BGR2LAB).astype(np.float32)
                                lab_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                                for ch in range(3):
                                    cf_ch = lab_cf[:, :, ch]
                                    orig_ch = lab_orig[:, :, ch]
                                    o_mean = orig_ch[mask_bool].mean()
                                    o_std = max(orig_ch[mask_bool].std(), 1e-6)
                                    c_mean = cf_ch[mask_bool].mean()
                                    c_std = max(cf_ch[mask_bool].std(), 1e-6)
                                    lab_cf[:, :, ch] = (cf_ch - c_mean) * (o_std / c_std) + o_mean
                                cf_matched = cv2.cvtColor(
                                    lab_cf.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                            else:
                                cf_matched = cf_full
                            result = (cf_matched.astype(np.float32) * mask_f +
                                      frame.astype(np.float32) * (1.0 - mask_f))
                            writer.write(result.astype(np.uint8))
                        else:
                            writer.write(cf_full)
                    else:
                        writer.write(cf_full)
                else:
                    writer.write(cf_full)
                count += 1
            except Exception as _cf_err:
                if idx < 3 or idx % 100 == 0:
                    logger.warning(f"  CodeFormer frame {idx} error: {_cf_err}")
                writer.write(frame)

            idx += 1
            if idx % 50 == 0:
                logger.info(f"  CodeFormer: {idx}/{total} (restored={count})")

    if count == 0 and idx > 0:
        logger.warning(f"  CodeFormer: 0/{idx} faces restored — net={'loaded' if net is not None else 'NONE'}")

    if mesh_ctx is not None:
        try:
            mesh_ctx.__exit__(None, None, None)
        except Exception:
            pass

    cap.release()
    writer.release()

    if count == 0:
        Path(temp).unlink(missing_ok=True)
        return False

    try:
        _ffmpeg_merge_audio(temp, input_video, output_video)
    except Exception as e:
        logger.error(f"  CodeFormer ffmpeg failed: {e}")
        Path(temp).unlink(missing_ok=True)
        return False

    Path(temp).unlink(missing_ok=True)
    logger.info(f"  CodeFormer: {count}/{total} frames (w={weight})")
    return True


def apply_temporal_smoothing(
    input_video: str, output_video: str, alpha: float = 0.25,
) -> bool:
    """Temporal inter-frame blending to reduce jitter."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp = str(Path(output_video).with_suffix(".smooth.mp4"))
    writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_frame = None
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is None:
            out = frame
        else:
            out = cv2.addWeighted(frame, alpha, prev_frame, 1.0 - alpha, 0)
        writer.write(out)
        prev_frame = frame
        count += 1

    cap.release()
    writer.release()

    if count == 0:
        Path(temp).unlink(missing_ok=True)
        return False

    try:
        _ffmpeg_merge_audio(temp, input_video, output_video)
    except Exception as e:
        logger.error(f"  Temporal smoothing ffmpeg failed: {e}")
        Path(temp).unlink(missing_ok=True)
        return False

    Path(temp).unlink(missing_ok=True)
    logger.info(f"  Temporal smoothing: {count}/{total} frames (alpha={alpha})")
    return True


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
async def health():
    """Health check with capability list."""
    caps = ["feathered_blend", "temporal_smoothing"]
    if GFPGAN_MODEL_PATH.exists():
        caps.append("gfpgan")
    if (CODEFORMER_WEIGHTS / "codeformer.pth").exists():
        caps.append("codeformer")
    return {"status": "ok", "capabilities": caps}


@app.post("/api/v1/postprocess")
async def postprocess(
    raw_video: UploadFile = File(..., description="Lip-synced output from ComfyUI"),
    original_video: UploadFile = File(..., description="Original input video for feathered blend"),
    enable_feathered_blend: bool = Form(True),
    feather_radius_px: int = Form(35),
    min_face_confidence: float = Form(0.6),
    full_blend_confidence: float = Form(0.85),
    enable_gfpgan: bool = Form(False),
    enable_codeformer: bool = Form(False),
    codeformer_weight: float = Form(0.5),
    codeformer_mouth_only: bool = Form(False),
    enable_temporal_smoothing: bool = Form(False),
    temporal_smooth_alpha: float = Form(0.25),
):
    """Apply post-processing pipeline to lip-synced video.

    Order: GFPGAN -> feathered blend -> CodeFormer -> temporal smoothing.
    Returns the final processed video file.
    """
    run_id = uuid.uuid4().hex[:8]
    run_dir = TEMP_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = str(run_dir / "raw.mp4")
    orig_path = str(run_dir / "original.mp4")
    current = raw_path

    try:
        # Save uploaded files
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(raw_video.file, f)
        with open(orig_path, "wb") as f:
            shutil.copyfileobj(original_video.file, f)

        logger.info(f"[{run_id}] Post-processing: gfpgan={enable_gfpgan}, "
                     f"feathered={enable_feathered_blend}, codeformer={enable_codeformer}, "
                     f"temporal={enable_temporal_smoothing}")

        # Step 1: GFPGAN (pre-blend)
        if enable_gfpgan:
            out = str(run_dir / "step1_gfpgan.mp4")
            if apply_gfpgan(current, out):
                current = out
                logger.info(f"[{run_id}] GFPGAN done")
            else:
                logger.warning(f"[{run_id}] GFPGAN skipped/failed")

        # Step 2: Feathered blend
        if enable_feathered_blend:
            out = str(run_dir / "step2_blend.mp4")
            if apply_feathered_blend(orig_path, current, out, feather_radius_px,
                                        min_face_confidence, full_blend_confidence):
                current = out
                logger.info(f"[{run_id}] Feathered blend done")
            else:
                logger.warning(f"[{run_id}] Feathered blend skipped/failed")

        # Step 3: CodeFormer (post-blend)
        if enable_codeformer:
            out = str(run_dir / "step3_codeformer.mp4")
            if apply_codeformer(current, out, codeformer_weight, codeformer_mouth_only):
                current = out
                logger.info(f"[{run_id}] CodeFormer done")
            else:
                logger.warning(f"[{run_id}] CodeFormer skipped/failed")

        # Step 4: Temporal smoothing
        if enable_temporal_smoothing:
            out = str(run_dir / "step4_smooth.mp4")
            if apply_temporal_smoothing(current, out, temporal_smooth_alpha):
                current = out
                logger.info(f"[{run_id}] Temporal smoothing done")
            else:
                logger.warning(f"[{run_id}] Temporal smoothing skipped/failed")

        logger.info(f"[{run_id}] Pipeline complete")
        return FileResponse(current, media_type="video/mp4", filename="postprocessed.mp4")

    except Exception as e:
        logger.error(f"[{run_id}] Pipeline failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Note: cleanup deferred — FastAPI streams the file before this runs
        pass


# ---------------------------------------------------------------------------
# Async post-process API (kolkata-diwali-alpana-arjun 2026-05-11, R40)
#
# The synchronous /api/v1/postprocess above streams the result back in the
# SAME HTTP response. For 6-second clips CodeFormer alone takes ~70s; total
# pipeline can hit 110-130s. RunPod's HTTP proxy enforces a ~100s read
# timeout, so the proxy returns 524 even though the sidecar is still
# working (and ultimately completes successfully). The client then falls
# back to the raw LatentSync output and skips face cleanup.
#
# These async endpoints decouple submit from result so each individual
# HTTP call returns in <1s — well under any proxy timeout.
#
#   POST /api/v1/postprocess/submit  → 202 { "job_id": "..." }
#     (background thread runs the pipeline)
#   GET  /api/v1/postprocess/status/{job_id} → { state: queued|running|done|failed, error?: }
#   GET  /api/v1/postprocess/result/{job_id} → streams the mp4 (only after state=done)
#
# In-memory job table — single-pod sidecar so no Redis needed.
# Old jobs auto-purged after RESULT_TTL_SEC to avoid filesystem fill.
# ---------------------------------------------------------------------------

_PP_JOBS: dict = {}
_PP_JOBS_LOCK = threading.Lock()
RESULT_TTL_SEC = 1800  # 30 min — generous, lipsync worker downloads within seconds


def _pp_purge_expired() -> None:
    """Drop in-memory job records older than RESULT_TTL_SEC and delete their files."""
    now = time.time()
    with _PP_JOBS_LOCK:
        stale = [jid for jid, r in _PP_JOBS.items()
                 if now - r.get("created_at", now) > RESULT_TTL_SEC]
        for jid in stale:
            r = _PP_JOBS.pop(jid, None)
            if r and r.get("run_dir"):
                shutil.rmtree(r["run_dir"], ignore_errors=True)


def _pp_pipeline_worker(job_id: str, run_dir: Path, raw_path: str, orig_path: str,
                        params: dict) -> None:
    """Run the same pipeline as the sync endpoint, but stash result in the
    job table instead of streaming it back. Errors land in the job table too
    so the client sees them via /status."""
    with _PP_JOBS_LOCK:
        _PP_JOBS[job_id]["state"] = "running"

    current = raw_path
    try:
        logger.info(f"[{job_id}] async pipeline start: gfpgan={params['enable_gfpgan']}, "
                    f"feathered={params['enable_feathered_blend']}, codeformer={params['enable_codeformer']}, "
                    f"temporal={params['enable_temporal_smoothing']}")

        if params["enable_gfpgan"]:
            out = str(run_dir / "step1_gfpgan.mp4")
            if apply_gfpgan(current, out):
                current = out
                logger.info(f"[{job_id}] GFPGAN done")
            else:
                logger.warning(f"[{job_id}] GFPGAN skipped/failed")

        if params["enable_feathered_blend"]:
            out = str(run_dir / "step2_blend.mp4")
            if apply_feathered_blend(orig_path, current, out,
                                      params["feather_radius_px"],
                                      params["min_face_confidence"],
                                      params["full_blend_confidence"]):
                current = out
                logger.info(f"[{job_id}] Feathered blend done")
            else:
                logger.warning(f"[{job_id}] Feathered blend skipped/failed")

        if params["enable_codeformer"]:
            out = str(run_dir / "step3_codeformer.mp4")
            if apply_codeformer(current, out,
                                 params["codeformer_weight"],
                                 params["codeformer_mouth_only"]):
                current = out
                logger.info(f"[{job_id}] CodeFormer done")
            else:
                logger.warning(f"[{job_id}] CodeFormer skipped/failed")

        if params["enable_temporal_smoothing"]:
            out = str(run_dir / "step4_smooth.mp4")
            if apply_temporal_smoothing(current, out, params["temporal_smooth_alpha"]):
                current = out
                logger.info(f"[{job_id}] Temporal smoothing done")
            else:
                logger.warning(f"[{job_id}] Temporal smoothing skipped/failed")

        logger.info(f"[{job_id}] async pipeline complete")
        with _PP_JOBS_LOCK:
            _PP_JOBS[job_id]["state"] = "done"
            _PP_JOBS[job_id]["output_path"] = current
    except Exception as e:
        logger.error(f"[{job_id}] async pipeline failed: {e}")
        with _PP_JOBS_LOCK:
            _PP_JOBS[job_id]["state"] = "failed"
            _PP_JOBS[job_id]["error"] = str(e)


@app.post("/api/v1/postprocess/submit")
async def postprocess_submit(
    raw_video: UploadFile = File(...),
    original_video: UploadFile = File(...),
    enable_feathered_blend: bool = Form(True),
    feather_radius_px: int = Form(35),
    min_face_confidence: float = Form(0.6),
    full_blend_confidence: float = Form(0.85),
    enable_gfpgan: bool = Form(False),
    enable_codeformer: bool = Form(False),
    codeformer_weight: float = Form(0.5),
    codeformer_mouth_only: bool = Form(False),
    enable_temporal_smoothing: bool = Form(False),
    temporal_smooth_alpha: float = Form(0.25),
):
    """Async submit — returns job_id immediately; client polls /status."""
    _pp_purge_expired()
    job_id = uuid.uuid4().hex[:12]
    run_dir = TEMP_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(run_dir / "raw.mp4")
    orig_path = str(run_dir / "original.mp4")
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(raw_video.file, f)
    with open(orig_path, "wb") as f:
        shutil.copyfileobj(original_video.file, f)

    params = {
        "enable_feathered_blend": enable_feathered_blend,
        "feather_radius_px": feather_radius_px,
        "min_face_confidence": min_face_confidence,
        "full_blend_confidence": full_blend_confidence,
        "enable_gfpgan": enable_gfpgan,
        "enable_codeformer": enable_codeformer,
        "codeformer_weight": codeformer_weight,
        "codeformer_mouth_only": codeformer_mouth_only,
        "enable_temporal_smoothing": enable_temporal_smoothing,
        "temporal_smooth_alpha": temporal_smooth_alpha,
    }

    with _PP_JOBS_LOCK:
        _PP_JOBS[job_id] = {
            "state": "queued",
            "created_at": time.time(),
            "run_dir": str(run_dir),
            "output_path": None,
            "error": None,
        }

    t = threading.Thread(
        target=_pp_pipeline_worker,
        args=(job_id, run_dir, raw_path, orig_path, params),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id, "state": "queued"}, status_code=202)


@app.get("/api/v1/postprocess/status/{job_id}")
async def postprocess_status(job_id: str):
    """Return job state. Polled by client every few seconds."""
    with _PP_JOBS_LOCK:
        rec = _PP_JOBS.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found (expired or never existed)")
    return {
        "job_id": job_id,
        "state": rec["state"],
        "error": rec.get("error"),
    }


@app.get("/api/v1/postprocess/result/{job_id}")
async def postprocess_result(job_id: str):
    """Stream the final processed mp4. Only valid when state=done."""
    with _PP_JOBS_LOCK:
        rec = _PP_JOBS.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    if rec["state"] != "done":
        raise HTTPException(status_code=409, detail=f"job state is {rec['state']!r} (need 'done')")
    return FileResponse(rec["output_path"], media_type="video/mp4", filename="postprocessed.mp4")


@app.on_event("shutdown")
def cleanup():
    """Clean up temp directory on shutdown."""
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
