import json
import sys
import argparse
import numpy as np
from snake_sim.render.core import FrameBuilder
from pathlib import Path
import cv2
import numpy as np


def frames_to_video(frames, output_abs_path, fps, size=None):
    """
    Converts a list of frames to a video file using opencv.
    The file ending has to be .mp4
    frames: list of frames, each frame is a numpy array with shape (height, width, 3) 3 being the RGB channels
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_abs_path), fourcc, fps, size)

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if size is not None:
            final_frame = cv2.resize(bgr_frame, size, interpolation=cv2.INTER_AREA)
        else:
            final_frame = bgr_frame
        out.write(final_frame)

    # Release everything when job is finished
    out.release()


def make_video(run_file, out_dir, fps, frames_multiply=1):
    print('making video for', run_file.stem)
    run_file_path = Path(run_file)
    filebasename = run_file_path.stem + '.mp4'
    output_path = Path(out_dir).joinpath(filebasename)
    frames = []
    with open(Path(run_file)) as run_file:
        run_data = json.load(run_file)
        metadata = run_data.copy()
        del metadata['steps']
        frame_builder = FrameBuilder(metadata, expand_factor=2, offset=(1, 1))
        for step_nr, step_data in run_data['steps'].items():
            subframes = frame_builder.step_to_frames(step_data)
            for frame in subframes:
                for _ in range(frames_multiply):
                    frames.append(frame)
    frames_to_video(frames, output_path, fps, size=(640, 640))

def cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str)
    args = ap.parse_args(argv)
    if not args.file:
        raise ValueError("No file specified")
    return args

def main(argv):
    args = cli(argv)
    script_location = Path(__file__).resolve().parent
    runs = script_location.joinpath('../runs/grid_32x32').resolve()
    output_dir = script_location.joinpath('videos').resolve()
    if args.file:
        run_file = Path(args.file)
        make_video(run_file, output_dir, fps=30, frames_multiply=2)
        return
    else:
        for run_file in runs.glob('*.json'):
            make_video(run_file, output_dir, fps=30, frames_multiply=2)
            break


if __name__ == '__main__':
    main(sys.argv[1:])