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
# reverse extra pixels so we're not right at edges
# and have room for labels
HORIZONTAL_MARGIN = 60
TOP_MARGIN = 100
BOTTOM_MARGIN = 80

# extra space for players outside baselines/sidelines
EXTRA_X = 3.0
EXTRA_Y = 3.5

# number of frames used for movement trails
TRAIL_FRAMES = 30

# how many pixels to use for every meter of tennis court
# take minimum of horizontal and vertical so we don't get cut off
SCALE = min(
    (WIDTH - 2 * HORIZONTAL_MARGIN) / (COURT_WIDTH + 2 * EXTRA_X),
    (HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) / (COURT_LENGTH + 2 * EXTRA_Y),
)

FAR_COLOR = (0, 165, 255)   # orange
NEAR_COLOR = (50, 230, 80)  # green
WHITE = (240, 240, 240)

def to_pixel(x_m, y_m):
    # convert court coordinates in meters to overhead image pixels and return
    x = WIDTH / 2 + (x_m - COURT_WIDTH / 2) * SCALE
    y = TOP_MARGIN + (y_m + EXTRA_Y) * SCALE
    return int(round(x)), int(round(y))

def draw_line(image, start, end, color=WHITE, thickness=3):\
    # draw line between 2 points on overhead tennis court
    cv2.line(image, to_pixel(*start), to_pixel(*end), color, thickness)

def make_court_background():
    # draw top-down tennis court using real dimensions
    # first draw green background
    image = np.full((HEIGHT, WIDTH, 3), (47, 105, 49), dtype=np.uint8)
    # fill in blue playing area
    cv2.rectangle(
        image, to_pixel(0, 0), to_pixel(COURT_WIDTH, COURT_LENGTH),
        (145, 95, 45), -1
    )

    # calculating singles sidelines and outer boundaries
    singles_left = (COURT_WIDTH - SINGLES_WIDTH) / 2
    singles_right = COURT_WIDTH - singles_left
    court_lines = [
        ((0, 0), (COURT_WIDTH, 0)),
        ((0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),
        ((0, 0), (0, COURT_LENGTH)),
        ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),
    ]
    # draw outer court boundaries
    for start, end in court_lines:
        draw_line(image, start, end)

    # draw singles sidelines
    for x in (singles_left, singles_right):
        draw_line(image, (x, 0), (x, COURT_LENGTH))
    # draw service lines
    for y in (NET_Y - SERVICE_DISTANCE, NET_Y + SERVICE_DISTANCE):
        draw_line(image, (singles_left, y), (singles_right, y))
    draw_line(
        image, 
        (COURT_WIDTH / 2, NET_Y - SERVICE_DISTANCE), 
        (COURT_WIDTH / 2, NET_Y + SERVICE_DISTANCE)
    )

    # draw net
    draw_line(image, (0, NET_Y), (COURT_WIDTH, NET_Y), (210, 210, 210), 4)
    # return finished court
    return image


def draw_trail(image, history, color):
    # draw line connecting player's recent positions to create trail
    positions = list(history)
    # loop through consecutive positions
    for previous, current in zip(positions[:-1], positions[1:]):
        # skip missing detections
        if previous is None or current is None:
            continue
        # draw segment of trail
        cv2.line(image, to_pixel(*previous), to_pixel(*current), color, 2)


def main():
    # take input for trajectory JSON file
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to trajectory JSON")
    args = parser.parse_args()
    input_path = Path(args.input)
    # load the trajectory data
    with open(input_path, "r") as file:
        data = json.load(file)
    fps = float(data["fps"])
    # make sure we have a valid fps value
    if fps <= 0:
        raise ValueError("FPS must be positive")

    # create output path and init video writer object
    output_path = input_path.with_name(f"{input_path.stem}_overhead.mp4")
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    # check if OpenCV successfully init video writer
    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    # draw out court background
    background = make_court_background()

    # store recent positions separately for both players in deques
    history = {
        "far": deque(maxlen=TRAIL_FRAMES),
        "near": deque(maxlen=TRAIL_FRAMES)
    }
    # assign colors for each player
    colors = {"far": FAR_COLOR, "near": NEAR_COLOR}
    # try/finally block to ensure we always release video writer
    try:
        # loop through all frames
        for frame in data["frames"]:
            # make a copy of the background so we don't modify original
            image = background.copy()
            # dict to hold current pos of far and near players
            positions = {}
            # loop through both players
            for side in ("far", "near"):
                # extract court coordinates for each player if they exist
                player = frame["players"][side]
                position = tuple(player["court_m"]) if player is not None else None
                # update the histories and current positions
                history[side].append(position)
                positions[side] = position
            # loop through both players
            for side in ("far", "near"):
                # draw movement trail
                draw_trail(image, history[side], colors[side])
                # if current player position is missing that's all we do
                position = positions[side]
                if position is None:
                    continue
                # otherwise, convert court coordinates to pixel coordinates
                x, y = to_pixel(*position)
                # draw player as filled in circle aligned with feet
                cv2.circle(image, (x, y), 9, colors[side], -1)
                cv2.circle(image, (x, y), 9, WHITE, 2)

            # timestamp and labels for current frame
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
                image, ("Kalman filtered positions" if "filter" in data else "Raw measurements (no smoothing)"),
                (30, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1
            )
            # writ ethe current video frame
            writer.write(image)
    # always release the video writer
    finally:
        writer.release()
    print(f"Saved overhead video: {output_path}")
    print(f"Frames: {len(data['frames'])} | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()