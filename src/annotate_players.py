import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from homography import load_homography, pixel_to_court

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISPLAY_WIDTH = 1200

# times when botsort reassigned ids in the first clip
REASSOCIATION_TIMES = [8.75, 11.65, 15.99, 24.23]


def select_frames(trajectory_data):
    # sample across video including trouble spots
    frames = trajectory_data["frames"]
    fps = float(trajectory_data["fps"])

    # sample 20 frames evenly after warmup
    selected = set(np.linspace(65, len(frames) - 10, 20, dtype=int).tolist())

    # include frames around id changes
    for time_s in REASSOCIATION_TIMES:
        index = round(time_s * fps)
        if 60 <= index < len(frames):
            selected.add(index)

    # include up to 3 frames with missing measurements
    missing = [
        f["frame"] for f in frames[60:]
        if f["players"]["far"] is None or f["players"]["near"] is None
    ]
    selected.update(missing[:3])

    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--trajectories", required=True)
    args = parser.parse_args()

    video_path = Path(args.video)

    with open(args.trajectories, "r") as f:
        trajectory_data = json.load(f)

    fps = float(trajectory_data["fps"])

    calibration_path = PROJECT_ROOT / "data" / "calibration" / f"{video_path.stem}.json"
    homography = load_homography(calibration_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(DISPLAY_WIDTH / width, 1.0)
    display_size = (round(width * scale), round(height * scale))

    selected_frames = select_frames(trajectory_data)
    annotations = []

    window_name = "Manual Player Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    clicks = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append([round(x / scale), round(y / scale)])

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        for sample_index, frame_index in enumerate(selected_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()

            if not success:
                print(f"could not read frame {frame_index}")
                continue

            clicks.clear()

            while True:
                display = cv2.resize(frame, display_size)

                # show selected ground contact points
                for index, point in enumerate(clicks):
                    x, y = round(point[0] * scale), round(point[1] * scale)
                    color = (0, 165, 255) if index == 0 else (0, 255, 0)
                    cv2.circle(display, (x, y), 6, color, -1)

                # instructions
                cv2.rectangle(display, (0, 0), (display_size[0], 85), (0, 0, 0), -1)

                if len(clicks) == 0:
                    instruction = "Click FAR player's feet (orange)"
                elif len(clicks) == 1:
                    instruction = "Click NEAR player's feet (green)"
                else:
                    instruction = "Press ENTER to save"

                cv2.putText(
                    display,
                    f"Frame {frame_index} | {frame_index / fps:.2f}s | {sample_index + 1}/{len(selected_frames)}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                cv2.putText(
                    display,
                    f"{instruction} | R: reset | S: skip | Q: quit",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("r"):
                    clicks.clear()
                elif key == ord("s"):
                    print(f"skipped frame {frame_index}")
                    break
                elif key == ord("q"):
                    print("stopping annotation early")
                    selected_frames = []
                    break
                elif key in (13, 10) and len(clicks) == 2:
                    players = {}

                    for side, point in zip(("far", "near"), clicks):
                        court_x, court_y = pixel_to_court(point[0], point[1], homography)
                        players[side] = {
                            "foot_pixel": point.copy(),
                            "court_m": [round(court_x, 3), round(court_y, 3)]
                        }

                    annotations.append({
                        "frame": frame_index,
                        "timestamp_s": round(frame_index / fps, 3),
                        "players": players
                    })

                    print(f"annotated frame {frame_index}")
                    break

            if not selected_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    output_dir = PROJECT_ROOT / "data" / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.json"

    with open(output_path, "w") as f:
        json.dump({"video": video_path.name, "annotations": annotations}, f, indent=2)

    print(f"\nsaved {len(annotations)} annotated frames")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()