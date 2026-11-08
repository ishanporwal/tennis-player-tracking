import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# standard doubles tennis court dimensions in meters
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
SINGLES_WIDTH = 8.23
SERVICE_LINE_DISTANCE = 6.40

# size for calibration window
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

# order to select 4 corners in
POINT_NAMES = ["Far-left", "Far-right", "Near-right", "Near-left"]


def draw_court_preview(frame, homography):
    # project standard tennis court lines onto the original frame

    # create copy of frame bc we draw court lines and don't
    # want to modify original
    preview = frame.copy()

    # compute matrix to convert coordinates from flat tennis court
    # to coordinates from camera's persepective (meters -> pixels)
    inverse_homography = np.linalg.inv(homography)

    net_y = COURT_LENGTH / 2
    singles_offset = (COURT_WIDTH - SINGLES_WIDTH) / 2

    # each line is defined using real-world court coordinates
    # each pair of points defines one straight line
    court_lines = [
        # doubles boundaries
        ((0, 0), (COURT_WIDTH, 0)),
        ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),
        ((COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH)),
        ((0, COURT_LENGTH), (0, 0)),

        # singles sidelines
        ((singles_offset, 0), (singles_offset, COURT_LENGTH)),
        ((COURT_WIDTH - singles_offset, 0), (COURT_WIDTH - singles_offset, COURT_LENGTH)),

        # service lines
        ((singles_offset, net_y - SERVICE_LINE_DISTANCE), (COURT_WIDTH - singles_offset, net_y - SERVICE_LINE_DISTANCE)),
        ((singles_offset, net_y + SERVICE_LINE_DISTANCE), (COURT_WIDTH - singles_offset, net_y + SERVICE_LINE_DISTANCE)),

        # center service line
        ((COURT_WIDTH / 2, net_y - SERVICE_LINE_DISTANCE), (COURT_WIDTH / 2, net_y + SERVICE_LINE_DISTANCE)),

        # net (ground-level court coordinates)
        ((0, net_y), (COURT_WIDTH, net_y)),
    ]

    # go through these court lines to transform to camera pixels 
    for start, end in court_lines:
        # package point for persepectiveTransform
        points = np.array([[start, end]], dtype=np.float32)
        # use inverse homography to figure out where to draw ccourt lines
        # in original camera image
        image_points = cv2.perspectiveTransform(points, inverse_homography)[0]

        # convert to integer pixel coordinate pairs
        # p1 = starting pixels for line, p2 = ending
        p1 = tuple(np.round(image_points[0]).astype(int))
        p2 = tuple(np.round(image_points[1]).astype(int))

        # actually drawing the green line
        cv2.line(preview, p1, p2, (0, 255, 0), 2)
    # return the modified image with green court lines
    return preview


def main():
    # allow choosing video + frame (in s) to calibrate with
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to tennis video")
    parser.add_argument("--time", type=float, default=3.0, help="Video timestamp used for calibration")
    # read command store values
    args = parser.parse_args()

    # read frame from video after going to timestamp
    video_path = Path(args.video).resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    success, frame = cap.read()
    # close video capture resource
    cap.release()
    # stop if frame couldn't be read
    if not success:
        raise RuntimeError("Could not read calibration frame")
    # image dimensions from frame read
    height, width = frame.shape[:2]

    # resize for display while retaining original coordinates
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    display_width = int(width * scale)
    display_height = int(height * scale)

    # list to store court corners user clicks
    selected_points = []
    # name of OpenCV window displaying tennis court frame
    window_name = "Court Calibration"

    def render():
        # resize the original image to fit the display window
        display = cv2.resize(frame, (display_width, display_height))

        # draw already selected corners
        for index, (x, y) in enumerate(selected_points):
            # again convert from original image coords to display's
            display_x = int(x * display_width / width)
            display_y = int(y * display_height / height)
            # draw filled green circle to represent it
            cv2.circle(display, (display_x, display_y), 6, (0, 255, 0), -1)
            cv2.putText(
                display, str(index + 1), (display_x + 10, display_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        # only after all 4 corners selected
        if len(selected_points) == 4:
            # convert each point into display coordinates and pack for OpenCV
            polygon = np.array([
                (int(x * display_width / width), int(y * display_height / height))
                for x, y in selected_points
            ], dtype=np.int32)
            # connect all 4 corners with green lines
            # closed polygon so last point connects back to first
            cv2.polylines(display, [polygon], True, (0, 255, 0), 2)

        # choose which instruction to display based on what's alr selected
        if len(selected_points) < 4:
            instruction = f"Click {len(selected_points) + 1}: {POINT_NAMES[len(selected_points)]}"
        else:
            instruction = "Press ENTER to save calibration"

        # draw filled black rectangle at top of window to make instructions
        # easy to read
        cv2.rectangle(display, (0, 0), (display_width, 40), (0, 0, 0), -1)
        cv2.putText(
            display, instruction, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        # return the completed display image
        return display

    def mouse_callback(event, x, y, _flags, _userdata):
        # ignore if we don't have a useful click or we're done
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(selected_points) >= 4:
            return

        # convert display coordinates back to original pixels
        # opposite of scaling done in render()
        original_x = x * width / display_width
        original_y = y * height / display_height

        # add selected corner post scaling to list
        selected_points.append([original_x, original_y])
        # print specific point and coordinates
        print(f"{POINT_NAMES[len(selected_points) - 1]}: ({original_x:.1f}, {original_y:.1f})")

    # create an OpenCV window with our name
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # register our callback function with this window
    cv2.setMouseCallback(window_name, mouse_callback)

    # keep calibration window open till user finishes/cancels
    while True:
        # render() creates display image with selected points
        # it's then displayed in the OpenCV window
        cv2.imshow(window_name, render())
        # wait 20ms for keyboard input
        key = cv2.waitKey(20) & 0xFF

        # key r resets the process
        if key == ord("r"):
            selected_points.clear()
            print("Selection reset")
        # escape or q closes the windows and exits
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            print("Calibration cancelled")
            return
        # if enter pressed + all 4 corners selected, exit
        elif key in (13, 10) and len(selected_points) == 4:
            break
    # close calibration window since corners are selected
    cv2.destroyAllWindows()

    # four selected points in original image pixel coordinates
    image_points = np.array(selected_points, dtype=np.float32)

    # corresponding real-world positions in meters
    court_points = np.array([
        [0, 0],
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH],
        [0, COURT_LENGTH],
    ], dtype=np.float32)

    # compute matrix to convert coordinates from cam persepective to
    # coordinates on flat tennis court (pixels -> meters)
    homography = cv2.getPerspectiveTransform(image_points, court_points)

    # save calibration
    calibration_dir = PROJECT_ROOT / "data" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / f"{video_path.stem}.json"

    # create dict with all calibration info  with lists for points
    calibration = {
        "video": video_path.name,
        "image_width": width,
        "image_height": height,
        "court_width_m": COURT_WIDTH,
        "court_length_m": COURT_LENGTH,
        "image_points": image_points.tolist(),
        "court_points": court_points.tolist(),
        "homography": homography.tolist(),
    }

    # write dict as JSON (can overwrite)
    with open(calibration_path, "w") as file:
        json.dump(calibration, file, indent=4)

    print(f"Calibration saved: {calibration_path}")
    print(f"Homography matrix:\n{homography}")

    # generate and save a visual preview for verification
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    preview = draw_court_preview(frame, homography)
    preview_path = output_dir / f"{video_path.stem}_calibration.png"
    # image write to output path
    cv2.imwrite(str(preview_path), preview)

    print(f"Calibration preview saved: {preview_path}")


if __name__ == "__main__":
    main()