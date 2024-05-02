import os
import json
import core
import sys
import argparse
import numpy as np
import subprocess
from render.core import FrameBuilder
from pathlib import Path
import cv2
import numpy as np


def make_video(run_file, out_dir, fps, frames_multiply=1):
    run_file_path = Path(run_file)
    size = (640, 640)
    filebasename = run_file_path.stem + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path(out_dir).joinpath(filebasename)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)

    with open(Path(run_file)) as run_file:
        run_data = json.load(run_file)
        metadata = run_data.copy()
        del metadata['steps']
        frame_builder = FrameBuilder(metadata, expand_factor=2, offset=(1, 1))
        for step_nr, step_data in run_data['steps'].items():
            subframes = frame_builder.step_to_frames(step_data)
            for frame in subframes:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                resized_frame_image = cv2.resize(bgr_frame, size, interpolation=cv2.INTER_AREA)
                for _ in range(frames_multiply):
                    out.write(resized_frame_image)

    # Release everything when job is finished
    out.release()

def cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=True)
    args = ap.parse_args(argv)
    if not args.file:
        raise ValueError("No file specified")
    return args

def main(argv):
    runs = Path('b:/pythonStuff/snake_sim/runs/grid_32x32').resolve()
    output_dir = Path('render/videos')
    for run_file in runs.glob('*.json'):
        print('making video for', run_file.stem)
        make_video(run_file, output_dir, fps=30, frames_multiply=2)


if __name__ == '__main__':
    main(sys.argv[1:])