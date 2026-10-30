from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_FRAMES = 1000

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError("No MP4 videos found in data/raw/")

    video_path = video_files[0]
    print(f"Processing: {video_path.name}")

    # open vid and get props
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output vid
    output_path = OUTPUT_DIR / f"{video_path.stem}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    # load yolo model and run on vid
    model = YOLO("yolo11n.pt")
    frame_count = 0
    try:
        while frame_count < MAX_FRAMES:
            success, frame = cap.read()
            if not success:
                break
            # detect ppl and associate them with existing tracks
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=0.25,
                device=0,
                verbose=False
            )

            boxes = results[0].boxes
            # draw bounding boxes around ppl detected
            if boxes is not None and boxes.id is not None:
                for box in boxes:
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    # colro based on id
                    color = (
                        (37 * track_id) % 256,
                        (97 * track_id) % 256,
                        (157 * track_id) % 256
                    )
                    # estimate feet loc of ppl
                    foot_x = (x1 + x2) // 2
                    foot_y = y2

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )
                    cv2.circle(
                        frame, (foot_x, foot_y), 5, color, -1
                    )
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
            # save and advance frame
            writer.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        cap.release()
        writer.release()
    print(f"Processed {frame_count} frames")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
