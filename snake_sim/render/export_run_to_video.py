import argparse
import os
import subprocess
import time
import queue
import threading
import wave
from importlib import resources
from pathlib import Path
from shutil import which

import numpy as np

from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.loop_observers.waitable_observer import WaitableObserver
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.environment.types import NoMoreSteps
from snake_sim.render.utils import create_color_map


AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2


with resources.open_text('snake_sim.render.sounds', 'eat.wav') as eat_sound_file:
    DEFAULT_EAT_SOUND_PATH = Path(eat_sound_file.name)


def _load_wav(path: Path) -> np.ndarray:
    """Load a WAV file and return float32 stereo samples at AUDIO_SAMPLE_RATE."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    samples /= np.iinfo(dtype).max

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    else:
        samples = samples.reshape(-1, 1)

    # Convert to stereo
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    else:
        samples = samples[:, :2]

    # Resample if needed (simple linear interp, good enough for SFX)
    if framerate != AUDIO_SAMPLE_RATE:
        ratio = AUDIO_SAMPLE_RATE / framerate
        new_len = int(len(samples) * ratio)
        old_idx = np.linspace(0, len(samples) - 1, new_len)
        left = np.interp(old_idx, np.arange(len(samples)), samples[:, 0])
        right = np.interp(old_idx, np.arange(len(samples)), samples[:, 1])
        samples = np.stack([left, right], axis=1)

    return samples.astype(np.float32)


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
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _read_gpu_vendors() -> set[str]:
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
    if "nvidia" in vendors and "h264_nvenc" in enc:
        return "h264_nvenc"
    if "intel" in vendors and "h264_qsv" in enc:
        return "h264_qsv"
    if _has_drm_render_node() and "h264_vaapi" in enc:
        return "h264_vaapi"
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
        return "p4"
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
    audio_pipe_path: str | None = None,
) -> subprocess.Popen:
    vf = None
    if out_width is not None and out_height is not None:
        if scale_backend == "cuda":
            if codec.strip() != "h264_nvenc":
                raise ValueError("scale_backend='cuda' is only supported with codec='h264_nvenc'")
            if output_pix_fmt not in {"yuv444p", "nv12", "yuv420p"}:
                raise ValueError(f"Unsupported output_pix_fmt for cuda scaling: {output_pix_fmt}")
            cuda_fmt = "yuv444p" if output_pix_fmt == "yuv444p" else "nv12"
            vf = (
                f"format={cuda_fmt},hwupload_cuda,"
                f"scale_cuda=w={out_width}:h={out_height}:interp_algo=nearest:format={cuda_fmt}"
            )
        else:
            vf = f"scale={out_width}:{out_height}:flags=neighbor"

    codec = codec.strip()
    if codec not in {"libx264", "h264_nvenc", "h264_vaapi", "h264_qsv"}:
        raise ValueError(
            f"Unsupported codec '{codec}'. Use auto|libx264|h264_nvenc|h264_vaapi|h264_qsv"
        )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", ffmpeg_loglevel]

    if threads is not None:
        cmd += ["-threads", str(int(threads))]
    if filter_threads is not None:
        cmd += ["-filter_threads", str(int(filter_threads))]

    # Video input from stdin
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{in_width}x{in_height}",
        "-r", str(fps),
        "-i", "-",
    ]

    # Audio input from named pipe (if provided)
    if audio_pipe_path is not None:
        cmd += [
            "-f", "f32le",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", str(AUDIO_CHANNELS),
            "-i", audio_pipe_path,
        ]

    if vf is not None:
        cmd += ["-vf", vf]

    if vf is not None and scale_backend == "cuda":
        cmd += ["-pix_fmt", "cuda"]

    cmd += ["-c:v", codec]

    if audio_pipe_path is not None:
        cmd += ["-c:a", "aac", "-ar", str(AUDIO_SAMPLE_RATE), "-b:a", "128k"]
    else:
        cmd += ["-an"]

    if preset is not None:
        cmd += ["-preset", str(preset)]

    if codec == "libx264":
        cmd += ["-crf", str(crf)]
        if output_pix_fmt == "yuv444p":
            cmd += ["-profile:v", "high444"]
    elif codec == "h264_nvenc":
        cmd += ["-rc:v", "vbr", "-cq:v", str(crf)]
        if output_pix_fmt == "yuv444p":
            cmd += ["-profile:v", "high444p"]
    elif codec == "h264_qsv":
        cmd += ["-global_quality", str(crf)]
    elif codec == "h264_vaapi":
        cmd += ["-qp", str(crf)]

    if scale_backend != "cuda":
        cmd += ["-pix_fmt", output_pix_fmt]

    cmd += [str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def export_run_to_video(
    run_path: Path,
    out_path: Path,
    fps: int = 30,
    max_width: int = 1080,
    max_height: int = 1080,
    max_size: int | None = None,
    expansion: int = 1,
    crf: int = 18,
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
    steps_per_second: float | None = None,
    start_step: int = 0,
    end_step: int | None = None,
    random_colors: bool = False,
    create_info_json: bool = False,
    eat_sound_path: Path | None = None,
):
    mode = "ffmpeg-scale" if scale_in_ffmpeg else "python-scale"
    preset_str = preset if preset is not None else "(none)"

    observable = FileRepeaterObservable(filepath=str(run_path))
    map_builder = MapBuilderObserver(expansion=expansion)
    state_builder = StateBuilderObserver()
    waitable = WaitableObserver()

    observable.add_observer(map_builder)
    observable.add_observer(state_builder)
    observable.add_observer(waitable)

    observable.start()
    waitable.wait_until_started()

    env_meta = map_builder.get_start_data().env_meta_data
    color_map = create_color_map(env_meta.snake_values, rand_colors=random_colors)
    food_value = env_meta.food_value

    while map_builder.get_max_step_idx() < start_step:
        time.sleep(0.01)
    first_map = map_builder.get_map_for_step(start_step)
    print(map_builder.get_current_step_idx())
    grid_h, grid_w = first_map.shape

    if max_size is not None:
        max_width = max_size
        max_height = max_size
    tile_px_w = max(1, max_width / grid_w)
    tile_px_h = max(1, max_height / grid_h)
    tile_px = min(tile_px_w, tile_px_h)
    out_w = grid_w * tile_px
    out_h = grid_h * tile_px

    if print_info:
        print(
            f"export_run_to_video: codec={codec} preset={preset_str} mode={mode} "
            f"scale_backend={scale_backend} fps={fps} grid={grid_w}x{grid_h} tile_px={tile_px} out={out_w}x{out_h}"
        )

    max_value = int(max(color_map.keys(), default=0))
    lut = _build_color_lut(color_map, max_value=max_value)

    if steps_per_second is None:
        steps_per_second = fps
    frames_per_step = fps / steps_per_second
    maps_per_step = expansion
    start_step = max(0, int(start_step))
    end_step = int(end_step) if end_step is not None else None

    # Load eat sound if provided
    eat_sound: np.ndarray | None = None
    if eat_sound_path is not None:
        eat_sound = _load_wav(eat_sound_path)
        if print_info:
            print(f"Loaded eat sound: {eat_sound_path} ({len(eat_sound)} samples at {AUDIO_SAMPLE_RATE}Hz)")

    # Set up named pipe for audio if we have a sound
    audio_pipe_path = None
    if eat_sound is not None:
        audio_pipe_path = f"/tmp/audio_pipe_{os.getpid()}"
        if os.path.exists(audio_pipe_path):
            os.remove(audio_pipe_path)
        os.mkfifo(audio_pipe_path)

    if scale_in_ffmpeg:
        in_w, in_h = grid_w, grid_h
        ff = _start_ffmpeg(
            out_path=out_path, in_width=in_w, in_height=in_h, fps=fps,
            codec=codec, crf=crf, preset=preset, out_width=out_w, out_height=out_h,
            ffmpeg_loglevel=ffmpeg_loglevel, threads=threads, filter_threads=filter_threads,
            scale_backend=scale_backend, output_pix_fmt=output_pix_fmt,
            audio_pipe_path=audio_pipe_path,
        )
    else:
        ff = _start_ffmpeg(
            out_path=out_path, in_width=out_w, in_height=out_h, fps=fps,
            codec=codec, crf=crf, preset=preset, out_width=None, out_height=None,
            ffmpeg_loglevel=ffmpeg_loglevel, threads=threads, filter_threads=filter_threads,
            scale_backend="cpu", output_pix_fmt=output_pix_fmt,
            audio_pipe_path=audio_pipe_path,
        )

    assert ff.stdin is not None

    frames_written = 0
    start_time = time.time()

    frame_queue: queue.Queue = queue.Queue(maxsize=100)
    # eat_queue carries True (ate food this frame) or False, or SENTINEL
    eat_queue: queue.Queue = queue.Queue(maxsize=200)
    SENTINEL = None

    def convert_map_to_frame(m: np.ndarray) -> np.ndarray:
        rgb = _map_to_rgb(m, lut)
        if scale_in_ffmpeg:
            return rgb
        else:
            return _scale_nearest(rgb, tile_px)

    def frame_producer():
        sim_step = start_step
        current_map_idx_floating = float(map_builder.get_current_map_idx())
        steps_with_audio = set()
        try:
            while True:
                if end_step is not None and sim_step >= end_step:
                    break
                try:
                    map_idx = int(current_map_idx_floating)
                    m = map_builder.get_map(map_idx)
                    sim_step = map_builder.get_current_step_idx()

                    frame = convert_map_to_frame(m)
                    frame_queue.put(frame)

                    # Detect food eating: only when there is one snake in the sim.

                    if eat_sound is not None:
                        only_one_snake = len(state_builder.get_start_data().env_meta_data.snake_values) == 1
                        current_state = state_builder.get_state(sim_step)
                        snake_ate = any(current_state.snake_ate.values())
                        add_sound = only_one_snake and snake_ate and sim_step not in steps_with_audio
                        eat_queue.put(add_sound)
                        steps_with_audio.add(sim_step)

                    current_map_idx_floating += (1 / frames_per_step) * maps_per_step
                except NoMoreSteps:
                    time.sleep(0.001)
                    continue
                except StopIteration:
                    break
                except queue.Full:
                    time.sleep(0.001)
                    continue
        finally:
            frame_queue.put(SENTINEL)
            if eat_sound is not None:
                eat_queue.put(SENTINEL)

    def frame_consumer():
        nonlocal frames_written
        while True:
            frame = frame_queue.get()
            if frame is SENTINEL:
                break
            ff.stdin.write(frame.tobytes(order="C"))
            frames_written += 1
            if progress_every and frames_written % progress_every == 0:
                elapsed = time.time() - start_time
                rate = frames_written / elapsed if elapsed > 0 else 0.0
                print(f"frames={frames_written} elapsed={elapsed:.1f}s rate={rate:.1f} fps")
        ff.stdin.close()

    def audio_producer():
        """Generate PCM audio stream, mixing eat sound at food eating frames."""
        samples_per_frame = AUDIO_SAMPLE_RATE / fps  # float, we'll round per frame
        sound_len = len(eat_sound)

        # Rolling buffer: holds pending audio to be mixed in future frames
        # Large enough to hold the full sound effect
        buffer_size = sound_len + int(samples_per_frame) * 4
        buffer = np.zeros((buffer_size, AUDIO_CHANNELS), dtype=np.float32)
        write_pos = 0  # position in buffer where next frame starts

        with open(audio_pipe_path, "wb") as pipe:
            frame_idx = 0
            accumulated = 0.0  # fractional sample accumulator

            while True:
                ate = eat_queue.get()
                if ate is SENTINEL:
                    break

                if ate:
                    end = write_pos + sound_len
                    if end <= buffer_size:
                        buffer[write_pos:end] += eat_sound
                    else:
                        # wrap around instead of clamping
                        space = buffer_size - write_pos
                        buffer[write_pos:] += eat_sound[:space]
                        remainder = end - buffer_size
                        buffer[:remainder] += eat_sound[space:]

                # How many samples for this frame
                accumulated += samples_per_frame
                n_samples = int(accumulated)
                accumulated -= n_samples

                # Write samples from buffer
                end = write_pos + n_samples
                if end <= buffer_size:
                    chunk = buffer[write_pos:end].copy()
                    buffer[write_pos:end] = 0.0  # clear after reading
                else:
                    # wrap around
                    part1 = buffer[write_pos:].copy()
                    buffer[write_pos:] = 0.0
                    remainder = end - buffer_size
                    part2 = buffer[:remainder].copy()
                    buffer[:remainder] = 0.0
                    chunk = np.concatenate([part1, part2])

                # Clamp to [-1, 1] to prevent clipping
                np.clip(chunk, -1.0, 1.0, out=chunk)
                pipe.write(chunk.astype(np.float32).tobytes())

                write_pos = end % buffer_size
                frame_idx += 1

    threads_list = [
        threading.Thread(target=frame_producer, daemon=True),
        threading.Thread(target=frame_consumer, daemon=True),
    ]
    if eat_sound is not None:
        # Audio producer must start before ffmpeg tries to open the pipe
        audio_thread = threading.Thread(target=audio_producer, daemon=True)
        audio_thread.start()

    for t in threads_list:
        t.start()
    for t in threads_list:
        t.join()

    if eat_sound is not None:
        audio_thread.join()
        try:
            os.remove(audio_pipe_path)
        except OSError:
            pass

    ret = ff.wait()
    elapsed = time.time() - start_time
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")

    print(f"Wrote {frames_written} frames to {out_path} ({elapsed:.2f}s)")

    if create_info_json:
        info = {
            "codec": codec,
            "preset": preset_str,
            "mode": mode,
            "scale_backend": scale_backend,
            "fps": fps,
            "grid_size": (grid_w, grid_h),
            "tile_px": tile_px,
            "out_size": (out_w, out_h),
            "total_frames": frames_written,
            "color_map": {k: color_map[k] for k in sorted(color_map.keys())},
            "steps_per_second": steps_per_second,
        }
        info_path = out_path.with_suffix(".json")
        import json
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Wrote export info to {info_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a .run or .run_proto file to an MP4 with correct grid aspect ratio (square tiles)"
    )
    ap.add_argument("run_file", type=Path, help="Input run file (.run or .run_proto)")
    ap.add_argument("out", type=Path, help="Output video file (e.g. out.mp4)")
    ap.add_argument("--steps-per-second", type=float, default=None)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--end-step", type=int, default=None)
    ap.add_argument("--fps", type=int, default=256)
    ap.add_argument("--max-width", type=int, default=1080)
    ap.add_argument("--max-height", type=int, default=1080)
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--expansion", type=int, default=2)
    ap.add_argument(
        "--codec", type=str, default="h264_nvenc",
        choices=["auto", "libx264", "h264_nvenc", "h264_vaapi", "h264_qsv"],
    )
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument(
        "--output-pix-fmt", type=str, default="yuv444p",
        choices=["yuv444p", "yuv420p"],
    )
    scale_group = ap.add_mutually_exclusive_group()
    scale_group.add_argument("--scale-in-ffmpeg", dest="scale_in_ffmpeg", action="store_true", default=True)
    scale_group.add_argument("--scale-in-python", dest="scale_in_ffmpeg", action="store_false")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--ffmpeg-loglevel", type=str, default="warning",
        choices=["quiet", "panic", "fatal", "error", "warning", "info"],
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--filter-threads", type=int, default=None)
    ap.add_argument(
        "--scale-backend", type=str, default="cuda",
        choices=["cpu", "cuda"],
    )
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--no-print-info", action="store_true", default=False)
    ap.add_argument("--random-colors", action="store_true", default=False)
    ap.add_argument("--create-info-json", action="store_true", default=False)
    ap.add_argument(
        "--eat-sound", type=Path, default=DEFAULT_EAT_SOUND_PATH,
        help="Path to a WAV file to play when the snake eats food",
    )

    args = ap.parse_args(argv)
    export_run_to_video(
        run_path=args.run_file,
        out_path=args.out,
        fps=args.fps,
        max_width=args.max_width,
        max_height=args.max_height,
        max_size=args.max_size,
        expansion=args.expansion,
        crf=args.crf,
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
        steps_per_second=args.steps_per_second,
        start_step=args.start_step,
        end_step=args.end_step,
        random_colors=args.random_colors,
        create_info_json=args.create_info_json,
        eat_sound_path=args.eat_sound,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())