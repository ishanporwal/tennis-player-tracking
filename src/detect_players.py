from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_FRAMES = 300

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))

    if not video_files:
        print("No mp4 vids found in data/raw")
        return

    video_path = video_files[0]
    print(f"Processing: {video_path.name}")

    # open vid and get props
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Couldn't open vid")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output vid
    output_path = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Couldn't create output vid")

    # load yolo model and run on vid
    model = YOLO("yolo11n.pt")
    frame_count = 0
    try:
        while frame_count < MAX_FRAMES:
            success, frame = cap.read()
            if not success:
                break
            # run yolo on cur frame
            results = model.predict(frame, 
                                    classes=[0], 
                                    conf=0.3, 
                                    device=0, 
                                    verbose=False)
            # draw bounding boxes around ppl detected
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            writer.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

    finally:
        cap.release()
        writer.release()

    print(f"Finished. Processed {frame_count} frames.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
