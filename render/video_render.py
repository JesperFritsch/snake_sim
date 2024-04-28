import os
import core
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=True)
    args = ap.parse_args(argv)
    if not args.file:
        raise ValueError("No file specified")

    filebasename = Path(args.file).stem
    video_file = os.path.join(os.getcwd(), 'videos', filebasename) + '.mp4'
    pixel_changes = core.pixel_changes_from_runfile(Path(args.file))
    height = pixel_changes['height']
    width = pixel_changes['width']
    target_size = (width*10, height*10)
    frame_arr = np.full((height, width, 3), pixel_changes['free_color'])
    frames = []
    # print(pixel_changes['height'])
    # print([x for x in pixel_changes])
    # print(pixel_changes['changes'][0])
    for change in pixel_changes['changes']:
        for pixel in change:
            (x, y), color = pixel
            frame_arr[y, x] = color
        frames.append(frame_arr.copy())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'XVID' to other codecs like 'MP4V', 'MJPG', etc.
    video = cv2.VideoWriter(video_file, fourcc, 20.0, target_size)

    for frame in frames:
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        video.write(resized_frame)
    video.release()



if __name__ == '__main__':
    main(sys.argv[1:])