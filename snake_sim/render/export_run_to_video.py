import argparse
import subprocess
import time
from pathlib import Path
from shutil import which
from typing import Optional

import numpy as np

from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.loop_observers.map_builder_observer import NoMoreSteps
from snake_sim.loop_observers.waitable_observer import WaitableObserver
from snake_sim.render.utils import create_color_map


def _build_color_lut(color_map: dict[int, tuple[int, int, int]], max_value: int) -> np.ndarray:
    if max_value < 0:
        raise ValueError("max_value must be >= 0")
    lut = np.zeros((max_value + 1, 3), dtype=np.uint8)
    for k, rgb in color_map.items():
        ik = int(k)
        if 0 <= ik <= max_value:
            lut[ik] = np.array(rgb, dtype=np.uint8)
    return lut


def _map_to_rgb(map_array: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # Fast path: direct LUT indexing (avoids per-frame np.unique + Python dict lookups)
    idx = map_array.astype(np.intp, copy=False)
    return lut[idx]


def _scale_nearest(rgb: np.ndarray, tile_px: int) -> np.ndarray:
    if tile_px <= 0:
        raise ValueError("tile_px must be > 0")
    return np.repeat(np.repeat(rgb, tile_px, axis=0), tile_px, axis=1)


def _ffmpeg_available_encoders() -> set[str]:
    if which("ffmpeg") is None:
        return set()
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return set()
    encoders: set[str] = set()
    for line in (p.stdout or "").splitlines():
        # lines look like: " V....D h264_nvenc ..."
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _read_gpu_vendors() -> set[str]:
    """Best-effort detection of GPU vendors present on the system.

    Returns a set containing any of: {'nvidia', 'intel', 'amd'}.
    """
    vendors: set[str] = set()
    drm_root = Path("/sys/class/drm")
    if not drm_root.exists():
        return vendors
    for vendor_file in drm_root.glob("card*/device/vendor"):
        try:
            vendor = vendor_file.read_text().strip().lower()
        except Exception:
            continue
        if vendor == "0x10de":
            vendors.add("nvidia")
        elif vendor == "0x8086":
            vendors.add("intel")
        elif vendor == "0x1002":
            vendors.add("amd")
    return vendors


def _has_drm_render_node() -> bool:
    return any(Path("/dev/dri").glob("renderD*") if Path("/dev/dri").exists() else [])


def _pick_default_codec(requested: str) -> str:
    requested = (requested or "").strip()
    if requested and requested != "auto":
        return requested

    enc = _ffmpeg_available_encoders()
    vendors = _read_gpu_vendors()

    # Prefer an encoder that matches the detected hardware.
    if "nvidia" in vendors and "h264_nvenc" in enc:
        return "h264_nvenc"
    if "intel" in vendors and "h264_qsv" in enc:
        return "h264_qsv"
    # VAAPI is commonly available for Intel/AMD via /dev/dri.
    if _has_drm_render_node() and "h264_vaapi" in enc:
        return "h264_vaapi"

    # Fallback: pick the first available hardware encoder, else CPU.
    for candidate in ("h264_nvenc", "h264_qsv", "h264_vaapi"):
        if candidate in enc:
            return candidate
    return "libx264"


def _pick_default_preset(codec: str, preset: str | None) -> str | None:
    if preset is not None and str(preset).strip() != "":
        return preset
    if codec == "libx264":
        return "veryfast"
    if codec == "h264_nvenc":
        # NVENC presets are typically p1..p7 (p1 fastest, p7 best)
        return "p4"
    # VAAPI/QSV don't consistently support -preset; omit by default
    return None


def _start_ffmpeg(
    out_path: Path,
    in_width: int,
    in_height: int,
    fps: int,
    codec: str,
    crf: int,
    preset: str | None,
    out_width: int | None = None,
    out_height: int | None = None,
    ffmpeg_loglevel: str = "warning",
    threads: int | None = None,
    filter_threads: int | None = None,
    scale_backend: str = "cpu",
    output_pix_fmt: str = "yuv444p",
) -> subprocess.Popen:
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it first (e.g. pacman -S ffmpeg).")

    if (out_width is None) != (out_height is None):
        raise ValueError("out_width and out_height must be both set or both None")

    # Keep tiles sharp if we scale in ffmpeg.
    # Default output_pix_fmt is yuv444p to avoid chroma-subsampling blur/bleed around sharp edges.
    vf = None
    if out_width is not None and out_height is not None:
        if scale_backend == "cuda":
            if codec.strip() != "h264_nvenc":
                raise ValueError("scale_backend='cuda' is only supported with codec='h264_nvenc'")
            # Requires hwupload_cuda + scale_cuda filters.
            # We keep frames on the GPU by outputting CUDA hwframes to the encoder.
            # (Still does RGB->NV12 conversion on CPU before upload.)
            # NOTE: using yuv444p avoids chroma blur. If this fails on some ffmpeg builds/drivers,
            # fall back to --scale-backend cpu.
            if output_pix_fmt not in {"yuv444p", "nv12", "yuv420p"}:
                raise ValueError(f"Unsupported output_pix_fmt for cuda scaling: {output_pix_fmt}")
            cuda_fmt = "yuv444p" if output_pix_fmt == "yuv444p" else "nv12"
            vf = (
                f"format={cuda_fmt},hwupload_cuda,"
                f"scale_cuda=w={out_width}:h={out_height}:interp_algo=nearest:format={cuda_fmt}"
            )
        else:
            vf = f"scale={out_width}:{out_height}:flags=neighbor"

    # Note:
    # - libx264 is CPU.
    # - h264_nvenc / h264_vaapi / h264_qsv use hardware encoders (GPU/iGPU).
    # - For vaapi you often need extra flags like -vaapi_device and hwupload; keep this simple for now.
    codec = codec.strip()
    if codec not in {"libx264", "h264_nvenc", "h264_vaapi", "h264_qsv"}:
        raise ValueError(
            f"Unsupported codec '{codec}'. Use auto|libx264|h264_nvenc|h264_vaapi|h264_qsv"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        ffmpeg_loglevel,
    ]

    if threads is not None:
        cmd += ["-threads", str(int(threads))]

    if filter_threads is not None:
        cmd += ["-filter_threads", str(int(filter_threads))]

    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{in_width}x{in_height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]

    if vf is not None:
        cmd += ["-vf", vf]

    # If we produced CUDA hwframes, make sure the encoder expects them
    if vf is not None and scale_backend == "cuda":
        cmd += ["-pix_fmt", "cuda"]

    # Encoder-specific settings
    cmd += ["-an", "-c:v", codec]

    # Preset (only if provided)
    if preset is not None:
        cmd += ["-preset", str(preset)]

    if codec == "libx264":
        cmd += ["-crf", str(crf)]
        if output_pix_fmt == "yuv444p":
            cmd += ["-profile:v", "high444"]
    elif codec == "h264_nvenc":
        # NVENC: use constant quality mode.
        cmd += ["-rc:v", "vbr", "-cq:v", str(crf)]
        if output_pix_fmt == "yuv444p":
            cmd += ["-profile:v", "high444p"]
    elif codec == "h264_qsv":
        # QSV: global_quality is the closest analogue.
        cmd += ["-global_quality", str(crf)]
    elif codec == "h264_vaapi":
        # VAAPI typically needs upload to GPU; we rely on ffmpeg's implicit hwupload if it can.
        # If this fails on your machine, pass --codec h264_nvenc or --codec libx264.
        cmd += ["-qp", str(crf)]

    # Output pixel format. yuv420p is most compatible but will blur sharp colored edges.
    # For the CUDA path we already set -pix_fmt cuda to feed hwframes to NVENC.
    if scale_backend != "cuda":
        cmd += ["-pix_fmt", output_pix_fmt]

    cmd += [str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def export_run_to_video(
    run_path: Path,
    out_path: Path,
    fps: int = 30,
    tile_px: int = 20,
    expansion: int = 1,
    crf: int = 18,
    max_frames: int | None = None,
    codec: str = "auto",
    preset: str | None = None,
    scale_in_ffmpeg: bool = True,
    ffmpeg_loglevel: str = "warning",
    print_info: bool = True,
    threads: int | None = None,
    filter_threads: int | None = None,
    scale_backend: str = "cpu",
    progress_every: int = 500,
    output_pix_fmt: str = "yuv444p",
):
    codec = _pick_default_codec(codec)
    preset = _pick_default_preset(codec, preset)
    if print_info:
        mode = "ffmpeg-scale" if scale_in_ffmpeg else "python-scale"
        preset_str = preset if preset is not None else "(none)"
        print(
            f"export_run_to_video: codec={codec} preset={preset_str} mode={mode} "
            f"scale_backend={scale_backend} fps={fps} tile_px={tile_px}"
        )
    observable = FileRepeaterObservable(filepath=str(run_path))
    map_builder = MapBuilderObserver(expansion=expansion)
    waitable = WaitableObserver()

    observable.add_observer(map_builder)
    observable.add_observer(waitable)

    observable.start()
    waitable.wait_until_started()

    # Wait until MapBuilder has received start data
    while map_builder._start_data is None:
        time.sleep(0.01)

    env_meta = map_builder._start_data.env_meta_data
    color_map = create_color_map(env_meta.snake_values)

    # First frame is the initial map created in notify_start
    first_map = map_builder.get_current_map()
    grid_h, grid_w = first_map.shape

    # Build LUT once (covers both map values and color-map keys)
    max_value = int(max(int(first_map.max()), max(color_map.keys(), default=0)))
    lut = _build_color_lut(color_map, max_value=max_value)

    # Validate color map coverage for the initial frame (cheap once)
    missing = set(np.unique(first_map).tolist()) - set(int(k) for k in color_map.keys())
    if missing:
        raise KeyError(f"No color mapping for tile values: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")

    # Two modes:
    # 1) scale_in_ffmpeg=False: Python scales up to final size and pushes huge frames (slow for big tile_px)
    # 2) scale_in_ffmpeg=True : Python pushes tiny grid frames; ffmpeg scales with nearest-neighbor (much faster I/O)
    if scale_in_ffmpeg:
        in_w, in_h = grid_w, grid_h
        out_w, out_h = grid_w * tile_px, grid_h * tile_px
        ff = _start_ffmpeg(
            out_path=out_path,
            in_width=in_w,
            in_height=in_h,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=preset,
            out_width=out_w,
            out_height=out_h,
            ffmpeg_loglevel=ffmpeg_loglevel,
            threads=threads,
            filter_threads=filter_threads,
            scale_backend=scale_backend,
            output_pix_fmt=output_pix_fmt,
        )
    else:
        out_w, out_h = grid_w * tile_px, grid_h * tile_px
        ff = _start_ffmpeg(
            out_path=out_path,
            in_width=out_w,
            in_height=out_h,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=preset,
            out_width=None,
            out_height=None,
            ffmpeg_loglevel=ffmpeg_loglevel,
            threads=threads,
            filter_threads=filter_threads,
            scale_backend="cpu",
            output_pix_fmt=output_pix_fmt,
        )

    assert ff.stdin is not None

    frames_written = 0
    start_time = time.time()

    def write_map(m: np.ndarray):
        nonlocal frames_written
        rgb = _map_to_rgb(m, lut)
        if scale_in_ffmpeg:
            frame = rgb
        else:
            frame = _scale_nearest(rgb, tile_px)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        ff.stdin.write(frame.tobytes(order="C"))
        frames_written += 1
        if progress_every and frames_written % progress_every == 0:
            elapsed = time.time() - start_time
            rate = frames_written / elapsed if elapsed > 0 else 0.0
            print(f"frames={frames_written} elapsed={elapsed:.1f}s rate={rate:.1f} fps")

    try:
        write_map(first_map)
        while True:
            if max_frames is not None and frames_written >= max_frames:
                break
            try:
                m = map_builder.get_next_map()
            except NoMoreSteps:
                # Producer hasn't delivered the next step yet.
                # If we already got stop_data, the next call will become StopIteration.
                time.sleep(0.001)
                continue
            except StopIteration:
                break
            write_map(m)
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass

    ret = ff.wait()
    elapsed = time.time() - start_time
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")

    print(f"Wrote {frames_written} frames to {out_path} ({elapsed:.2f}s)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a .run or .run_proto file to an MP4 with correct grid aspect ratio (square tiles)"
    )
    ap.add_argument("run_file", type=Path, help="Input run file (.run or .run_proto)")
    ap.add_argument("out", type=Path, help="Output video file (e.g. out.mp4)")
    # Defaults are intentionally tuned for this repository's typical use on this machine:
    # - high-FPS exports
    # - ffmpeg-side scaling
    # - NVIDIA NVENC + CUDA scaling
    ap.add_argument("--fps", type=int, default=256)
    ap.add_argument("--tile-px", type=int, default=30, help="Pixels per grid cell (square)")
    ap.add_argument("--expansion", type=int, default=2, help="MapBuilderObserver expansion factor")
    ap.add_argument(
        "--codec",
        type=str,
        default="h264_nvenc",
        choices=["auto", "libx264", "h264_nvenc", "h264_vaapi", "h264_qsv"],
        help="Video encoder. Default is h264_nvenc on this machine; 'auto' picks the best available (nvenc/vaapi/qsv/libx264).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Encoder preset override (default depends on codec; e.g. libx264=veryfast, nvenc=p4)",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Quality control. For libx264 this is CRF (18-28 typical). For hw encoders this is mapped to cq.",
    )
    ap.add_argument(
        "--output-pix-fmt",
        type=str,
        default="yuv444p",
        choices=["yuv444p", "yuv420p"],
        help="Output pixel format. Use yuv444p (default) for crisp colored tile edges; yuv420p for maximum compatibility.",
    )
    scale_group = ap.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--scale-in-ffmpeg",
        dest="scale_in_ffmpeg",
        action="store_true",
        default=True,
        help="(Default) Send small grid frames to ffmpeg and let ffmpeg do nearest-neighbor scaling.",
    )
    scale_group.add_argument(
        "--scale-in-python",
        dest="scale_in_ffmpeg",
        action="store_false",
        help="Scale frames in Python (slow for large outputs; mostly for debugging).",
    )
    ap.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    ap.add_argument(
        "--ffmpeg-loglevel",
        type=str,
        default="warning",
        choices=["quiet", "panic", "fatal", "error", "warning", "info"],
        help="ffmpeg log verbosity (use info to confirm which encoder is actually used)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ffmpeg worker threads (-threads). For libx264 this can help; for NVENC it mostly affects filters.",
    )
    ap.add_argument(
        "--filter-threads",
        type=int,
        default=None,
        help="ffmpeg filter threads (-filter_threads). Useful if scaling/filtering is CPU-bound.",
    )
    ap.add_argument(
        "--scale-backend",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Scaling backend when using --scale-in-ffmpeg. Use 'cuda' to offload scaling via scale_cuda (best with h264_nvenc).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N frames (0 disables).",
    )
    ap.add_argument(
        "--no-print-info",
        action="store_true",
        default=False,
        help="Don't print the selected codec/preset/mode line",
    )

    args = ap.parse_args(argv)
    export_run_to_video(
        run_path=args.run_file,
        out_path=args.out,
        fps=args.fps,
        tile_px=args.tile_px,
        expansion=args.expansion,
        crf=args.crf,
        max_frames=args.max_frames,
        codec=args.codec,
        preset=args.preset,
        scale_in_ffmpeg=args.scale_in_ffmpeg,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
        print_info=(not args.no_print_info),
        threads=args.threads,
        filter_threads=args.filter_threads,
        scale_backend=args.scale_backend,
        progress_every=args.progress_every,
        output_pix_fmt=args.output_pix_fmt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
