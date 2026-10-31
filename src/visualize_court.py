import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# standard tennis court dimensions (meters)
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
SINGLES_WIDTH = 8.23
NET_Y = COURT_LENGTH / 2
SERVICE_DISTANCE = 6.40

# output video dimensions
WIDTH = 720
HEIGHT = 1280
HORIZONTAL_MARGIN = 60
TOP_MARGIN = 100
BOTTOM_MARGIN = 80

# extra space for players outside baselines/sidelines
EXTRA_X = 3.0
EXTRA_Y = 3.5

# number of frames used for movement trails
TRAIL_FRAMES = 30

SCALE = min(
    (WIDTH - 2 * HORIZONTAL_MARGIN) / (COURT_WIDTH + 2 * EXTRA_X),
    (HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) / (COURT_LENGTH + 2 * EXTRA_Y),
)

FAR_COLOR = (0, 165, 255) # orange
NEAR_COLOR = (50, 230, 80) # green
WHITE = (240, 240, 240)

def to_pixel(x_m, y_m):
    # convert court coordinates in meters to overhead image pixels
    x = WIDTH / 2 + (x_m - COURT_WIDTH / 2) * SCALE
    y = TOP_MARGIN + (y_m + EXTRA_Y) * SCALE
    return int(round(x)), int(round(y))

def draw_line(image, start, end, color=WHITE, thickness=3):
    cv2.line(image, to_pixel(*start), to_pixel(*end), color, thickness)

def make_court_background():
    # draw top-down tennis court using real dimensions
    image = np.full((HEIGHT, WIDTH, 3), (47, 105, 49), dtype=np.uint8)
    # fill playing area
    cv2.rectangle(
        image, to_pixel(0, 0), to_pixel(COURT_WIDTH, COURT_LENGTH),
        (145, 95, 45), -1
    )

    singles_left = (COURT_WIDTH - SINGLES_WIDTH) / 2
    singles_right = COURT_WIDTH - singles_left

    # outer doubles boundaries
    court_lines = [
        ((0, 0), (COURT_WIDTH, 0)),
        ((0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),
        ((0, 0), (0, COURT_LENGTH)),
        ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),
    ]

    for start, end in court_lines:
        draw_line(image, start, end)

    # singles sidelines
    for x in (singles_left, singles_right):
        draw_line(image, (x, 0), (x, COURT_LENGTH))
    # service lines
    for y in (NET_Y - SERVICE_DISTANCE, NET_Y + SERVICE_DISTANCE):
        draw_line(image, (singles_left, y), (singles_right, y))
    # center service line
    draw_line(
        image, 
        (COURT_WIDTH / 2, NET_Y - SERVICE_DISTANCE), 
        (COURT_WIDTH / 2, NET_Y + SERVICE_DISTANCE)
    )

    # net
    draw_line(image, (0, NET_Y), (COURT_WIDTH, NET_Y), (210, 210, 210), 4)
    return image


def draw_trail(image, history, color):
    # draw recent measured positions without bridging missing frames
    positions = list(history)

    for previous, current in zip(positions[:-1], positions[1:]):
        if previous is None or current is None:
            continue
        cv2.line(image, to_pixel(*previous), to_pixel(*current), color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to trajectory JSON")
    args = parser.parse_args()
    input_path = Path(args.input)
    with open(input_path, "r") as file:
        data = json.load(file)
    fps = float(data["fps"])
    if fps <= 0:
        raise ValueError("FPS must be positive")

    output_path = input_path.parent / f"{Path(data['video']).stem}_overhead.mp4"

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    background = make_court_background()

    # store recent positions separately for both players
    history = {
        "far": deque(maxlen=TRAIL_FRAMES),
        "near": deque(maxlen=TRAIL_FRAMES)
    }

    colors = {"far": FAR_COLOR, "near": NEAR_COLOR}

    try:
        for frame in data["frames"]:
            image = background.copy()
            positions = {}
            # read player positions for this frame
            for side in ("far", "near"):
                player = frame["players"][side]
                position = tuple(player["court_m"]) if player is not None else None
                
                history[side].append(position)
                positions[side] = position
            # draw trajectories and current player positions
            for side in ("far", "near"):
                draw_trail(image, history[side], colors[side])
                position = positions[side]
                if position is None:
                    continue
                x, y = to_pixel(*position)
                cv2.circle(image, (x, y), 9, colors[side], -1)
                cv2.circle(image, (x, y), 9, WHITE, 2)

            # timestamp and labels
            timestamp = frame["timestamp_s"]

            cv2.putText(
                image, f"Tennis court positions | {timestamp:.2f}s",
                (30, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2
            )
            cv2.putText(
                image, "Orange: Far player   Green: Near player",
                (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1
            )
            cv2.putText(
                image, "Raw measurements (no smoothing)",
                (30, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1
            )

            writer.write(image)

    finally:
        writer.release()

    print(f"Saved overhead video: {output_path}")
    print(f"Frames: {len(data['frames'])} | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()