from collections import defaultdict
from math import hypot
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

WARMUP_FRAMES = 60
MIN_OBSERVATIONS = 10

COURT_ROI = [
    (0.40, 0.07),
    (0.62, 0.07),
    (0.78, 0.48),
    (0.98, 0.98),
    (0.02, 0.98),
    (0.22, 0.48),
]

NET_Y_RATIO = 0.45

def get_detections(boxes):
    # extract track ids, bounding boxes, and foot positions
    detections = []
    if boxes is None or boxes.id is None:
        return detections
    for box in boxes:
        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "id": track_id,
            "bbox": (x1, y1, x2, y2),
            "foot": ((x1 + x2) // 2, y2),
        })
    return detections

def identify_players(counts, last_observations):
    # select one persistent track per side of the court
    states = {}
    for side in ("far", "near"):
        if not counts[side]:
            raise RuntimeError(f"No candidates found on {side} side")
        track_id = max(counts[side], key=counts[side].get)
        if counts[side][track_id] < MIN_OBSERVATIONS:
            raise RuntimeError(f"Not enough observations for {side} player")
        foot, last_seen = last_observations[track_id]
        states[side] = {
            "id": track_id,
            "foot": foot,
            "last_seen": last_seen,
        }
        print(f"Initial {side} player: ID {track_id}")
    return states

def reassociate_players(states, detections, frame_index, fps, height, net_y, roi):
    # recover player identities when botsort changes ids
    active_tracks = {d["id"]: d for d in detections}

    # update players whose original ids are still active
    for side, state in states.items():
        detection = active_tracks.get(state["id"])
        if detection is not None:
            state["foot"] = detection["foot"]
            state["last_seen"] = frame_index

    # try to recover missing players
    for side, state in states.items():
        if state["id"] in active_tracks:
            continue
        gap = frame_index - state["last_seen"]
        # avoid reassociating from a very old position
        if gap > max(1, int(fps * 2)):
            continue

        # allow more movement for the larger near-side player
        base_distance = height * (0.11 if side == "far" else 0.18)
        max_distance = base_distance + height * 0.003 * min(gap, 20)
        used_ids = {s["id"] for s in states.values() if s["id"] in active_tracks}
        best_detection = None
        best_score = float("inf")

        for detection in detections:
            if detection["id"] in used_ids:
                continue
            foot_x, foot_y = detection["foot"]
            # candidates must remain on the correct court half
            candidate_side = "far" if foot_y < net_y else "near"
            if candidate_side != side:
                continue

            previous_x, previous_y = state["foot"]
            distance = hypot(foot_x - previous_x, foot_y - previous_y)
            if distance > max_distance:
                continue

            # prefer candidates inside roi without requiring it
            inside_roi = cv2.pointPolygonTest(roi, (float(foot_x), float(foot_y)), False) >= 0
            score = distance
            if not inside_roi:
                score += height * 0.06

            if score < best_score:
                best_score = score
                best_detection = detection

        if best_detection is not None:
            old_id = state["id"]
            new_id = best_detection["id"]
            state["id"] = new_id
            state["foot"] = best_detection["foot"]
            state["last_seen"] = frame_index
            print(f"{side.capitalize()} player reassociated: ID {old_id} -> ID {new_id} at {frame_index / fps:.2f}s")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError("No MP4 videos in data/raw/")

    video_path = video_files[0]
    print(f"Processing: {video_path.name}")

    # open vid and get props
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = np.array([(int(x * width), int(y * height)) for x, y in COURT_ROI], dtype=np.int32)
    net_y = int(height * NET_Y_RATIO)

    # output vid
    output_path = OUTPUT_DIR / f"{video_path.stem}_players.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    # load yolo model and run on vid
    model = YOLO("yolo11n.pt")
    counts = {"far": defaultdict(int), "near": defaultdict(int)}
    last_observations = {}
    states = None
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # detect ppl and associate them with existing tracks
            results = model.track(
                frame, persist=True, tracker="botsort.yaml", classes=[0],
                conf=0.25, device=0, verbose=False
            )
            detections = get_detections(results[0].boxes)

            if frame_index < WARMUP_FRAMES:
                for detection in detections:
                    track_id = detection["id"]
                    foot_x, foot_y = detection["foot"]
                    last_observations[track_id] = (detection["foot"], frame_index)
                    
                    inside = cv2.pointPolygonTest(roi, (float(foot_x), float(foot_y)), False)
                    if inside >= 0:
                        side = "far" if foot_y < net_y else "near"
                        counts[side][track_id] += 1

                cv2.putText(
                    frame, "Identifying players...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
            else:
                if states is None:
                    states = identify_players(counts, last_observations)
                
                reassociate_players(states, detections, frame_index, fps, height, net_y, roi)
                active_tracks = {d["id"]: d for d in detections}

                # draw bounding boxes around ppl detected
                for side, state in states.items():
                    detection = active_tracks.get(state["id"])
                    if detection is None:
                        continue

                    x1, y1, x2, y2 = detection["bbox"]
                    foot_x, foot_y = detection["foot"]
                    
                    # colro based on side
                    color = ((0, 165, 255) if side == "far" else (0, 255, 0))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # estimate feet loc of ppl
                    cv2.circle(frame, (foot_x, foot_y), 5, color, -1)
                    
                    cv2.putText(
                        frame, f"{side.capitalize()} Player ID {state['id']}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

            cv2.polylines(frame, [roi], True, (255, 255, 0), 2)
            
            # save and advance frame
            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames")

    finally:
        cap.release()
        writer.release()

    print(f"Processed {frame_index} frames")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()