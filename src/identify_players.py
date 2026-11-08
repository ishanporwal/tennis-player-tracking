from collections import defaultdict
from math import hypot
from pathlib import Path

import argparse
import json
import cv2
import numpy as np
from ultralytics import YOLO
from homography import load_homography, pixel_to_court

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# use first 60 frames to determine whcih detected
# person is each tennis player
WARMUP_FRAMES = 60
MIN_OBSERVATIONS = 10

# general court region for focusing on players
COURT_ROI = [
    (0.40, 0.07),
    (0.62, 0.07),
    (0.78, 0.48),
    (0.98, 0.98),
    (0.02, 0.98),
    (0.22, 0.48),
]

# approx where the net divides the image vertically
NET_Y_RATIO = 0.45

def get_detections(boxes):
    # extract track ids, bounding boxes, and foot positions
    detections = []
    # check if no bounding boxes provided
    if boxes is None or boxes.id is None:
        return detections
    for box in boxes:
        # get the tracking id assigned by BoT-SORT
        track_id = int(box.id.item())
        # get the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # store this detection
        detections.append({
            "id": track_id,
            "bbox": (x1, y1, x2, y2),
            # foot = horizontal center of bounding box
            "foot": ((x1 + x2) // 2, y2),
        })
    return detections

def identify_players(counts, last_observations):
    # select one persistent track per side of the court
    states = {}
    # figure out likely id for player on each side
    for side in ("far", "near"):
        # no potential players found on either side of court
        if not counts[side]:
            raise RuntimeError(f"No candidates found on {side} side")
        # select most frequently observed tracked id
        track_id = max(counts[side], key=counts[side].get)
        # require enough observations to be certain
        if counts[side][track_id] < MIN_OBSERVATIONS:
            raise RuntimeError(f"Not enough observations for {side} player")
        # get the player's last position and initialize player state
        foot, last_seen = last_observations[track_id]
        states[side] = {
            "id": track_id,
            "foot": foot,
            # "last_seen": last_seen, frame idx most recently detected
        }
        print(f"Initial {side} player: ID {track_id}")
    return states

def reassociate_players(states, detections, frame_index, fps, height, net_y, roi):
    # recover player ids when BoT-SORT changes ids

    # create dict searchable by id
    active_tracks = {d["id"]: d for d in detections}

    # update players whose original ids are still active
    for side, state in states.items():
        # look up if id was detected in this frame
        detection = active_tracks.get(state["id"])
        # if it was detected then update its location and last seen
        if detection is not None:
            state["foot"] = detection["foot"]
            state["last_seen"] = frame_index

    # try to recover missing players, run per side
    for side, state in states.items():
        # skip player if id still there in this frame
        if state["id"] in active_tracks:
            continue
        # calc frames passed since player last observed
        gap = frame_index - state["last_seen"]
        # avoid reassociating from a very old position
        if gap > max(1, int(fps * 2)):
            continue

        # allow more movement allowed for the larger near-side player
        base_distance = height * (0.11 if side == "far" else 0.18)
        # use percentages of video height to figure out reasonable max movement 
        max_distance = base_distance + height * 0.003 * min(gap, 20)
        # get tracking ids alr assigned to players successfully found so we skip
        used_ids = {s["id"] for s in states.values() if s["id"] in active_tracks}
        # best suitable replacement detection info
        best_detection = None
        # try to find best option with lowest score so init to infinity
        # score = distance + roi penalty
        best_score = float("inf")

        # go through all people detected in current frame
        for detection in detections:
            # if id already assigned to active player, skip
            if detection["id"] in used_ids:
                continue
            foot_x, foot_y = detection["foot"]
            # candidates must remain on the correct court half
            candidate_side = "far" if foot_y < net_y else "near"
            # reject candidates on the opposite side
            if candidate_side != side:
                continue
            # calc euclidean distance from the last known pos
            previous_x, previous_y = state["foot"]
            distance = hypot(foot_x - previous_x, foot_y - previous_y)
            # reject candidate if they're too far from last known pos
            if distance > max_distance:
                continue

            # prefer candidates inside roi without requiring it
            inside_roi = cv2.pointPolygonTest(roi, (float(foot_x), float(foot_y)), False) >= 0
            score = distance
            # penalty if outside roi
            if not inside_roi:
                score += height * 0.06
            # update best detection based on lowest score
            if score < best_score:
                best_score = score
                best_detection = detection
        # reassociate the missing player by updating state
        if best_detection is not None:
            old_id = state["id"]
            new_id = best_detection["id"]
            state["id"] = new_id
            state["foot"] = best_detection["foot"]
            state["last_seen"] = frame_index
            print(f"{side.capitalize()} player reassociated: ID {old_id} -> ID {new_id} at {frame_index / fps:.2f}s")


def main():
    # intialize output directory and video selection
    OUTPUT_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=None,
        help="Name of the input video in data/raw/"
    )
    args = parser.parse_args()
    # try to find the selected video
    # if not found, select first MP4 in alphabetical order
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError("No MP4 videos found in data/raw/")
    video_path = (
        VIDEO_DIR / args.video
        if args.video
        else video_files[0]
    )
    # check whether the path exists and is a regular file
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"Processing: {video_path.name}")
    
    # load homography for vid
    calibration_path = (
        PROJECT_ROOT / "data" / "calibration" / f"{video_path.stem}.json"
    )
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"No calibration found for {video_path.name}"
        )
    homography = load_homography(calibration_path)
    print(f"Loaded calibration: {calibration_path.name}")

    # open vid and get props
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    # fall back to 30 fps
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # iterate through court roi coordinates and scale for specific vid
    roi = np.array([(int(x * width), int(y * height)) for x, y in COURT_ROI], dtype=np.int32)
    net_y = int(height * NET_Y_RATIO)

    # output vid to save annotated vid
    output_path = OUTPUT_DIR / f"{video_path.stem}_players.mp4"
    # videowriter writes frames into output vid
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # if OpenCV couldn't init output video writer, release and exit
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    # create object and load pretrained YOLO model into memory
    model = YOLO("yolo11n.pt")
    # dict to count which ids identified on which side of the cour the most
    counts = {"far": defaultdict(int), "near": defaultdict(int)}
    # dict to store latest observed foot pos and frame idx for each id
    last_observations = {}
    # later becomes dict with player info
    states = None
    frame_index = 0
    # list to store player-position records for every frame
    trajectory_frames = []

    # start processing frames
    # try/finally block to ensure we clean up
    try:
        while True:
            # get next frame from vid
            success, frame = cap.read()
            # if end of vid, break
            if not success:
                break

            # detect ppl (class 0) and associate them with existing tracks using Bot-SORT
            results = model.track(
                frame, persist=True, tracker="botsort.yaml", classes=[0],
                conf=0.25, device=0, verbose=False
            )
            # make the detections we get easier to access -> id,bbox,foot
            detections = get_detections(results[0].boxes)

            # initially try to figure out which id belongs to near and far players
            # only run this during the first few frames
            if frame_index < WARMUP_FRAMES:
                # go through each tracked person in this frame
                for detection in detections:
                    track_id = detection["id"]
                    foot_x, foot_y = detection["foot"]
                    # store the latest observation for this id (overwriting)
                    last_observations[track_id] = (detection["foot"], frame_index)
                    # check whether the person is in the court roi by testing if it's in the poly
                    inside = cv2.pointPolygonTest(roi, (float(foot_x), float(foot_y)), False)
                    # if we're either on the boundry or within the roi
                    if inside >= 0:
                        # foot above net = far player, foot below net = near player
                        side = "far" if foot_y < net_y else "near"
                        # update observation counts
                        counts[side][track_id] += 1
                cv2.putText(
                    frame, "Identifying players...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
            else:
                # iniital warmup is done so identify the players
                if states is None:
                    states = identify_players(counts, last_observations)

                # reasociate players in case their id changed or dropped
                reassociate_players(states, detections, frame_index, fps, height, net_y, roi)
                
                # create local dictionary searchable by id
                active_tracks = {d["id"]: d for d in detections}

                # loop through the 2 players stored in states
                for side, state in states.items():
                    # get the current detection associated with that player
                    detection = active_tracks.get(state["id"])
                    # if the player isn't detected in this frame, skip bbox drawing
                    if detection is None:
                        continue
                    # extract bbox and foot coords
                    x1, y1, x2, y2 = detection["bbox"]
                    foot_x, foot_y = detection["foot"]
                    
                    # color based on side
                    color = ((0, 165, 255) if side == "far" else (0, 255, 0))
                    # draw the bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # estimate feet loc of ppl
                    cv2.circle(frame, (foot_x, foot_y), 5, color, -1)
                    # label the player
                    cv2.putText(
                        frame, f"{side.capitalize()} Player ID {state['id']}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
            # draw roi on every single frame including warmup
            cv2.polylines(frame, [roi], True, (255, 255, 0), 2)

            # create one trajectory record per video frame
            frame_record = {
                "frame": frame_index,
                "timestamp_s": round(frame_index / fps, 3),
                "players": {
                    "far": None,
                    "near": None
                }
            }
            # only record player pos after initial identification is done
            if states is not None:
                # reconstruct the detection dict by id
                active_tracks = {detection["id"]: detection for detection in detections}
                # loop through both players
                for side, state in states.items():
                    detection = active_tracks.get(state["id"])
                    # don't record position if player missing
                    if detection is None:
                        continue
                    # estimated grnd contact position in pixels
                    foot_x, foot_y = detection["foot"]
                    # convert cam pixels to court meters using homography transformation matrix
                    court_x, court_y = pixel_to_court(foot_x, foot_y, homography)
                    # store this specific position
                    frame_record["players"][side] = {
                        "track_id": state["id"],
                        "foot_pixel": [foot_x, foot_y],
                        "court_m": [
                            round(court_x, 3),
                            round(court_y, 3)
                        ]
                    }
            # add cur frame's dict to frame list
            trajectory_frames.append(frame_record)
            # save and advance frame
            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames")

    finally:
        # release the input capture resource and close the output writer
        cap.release()
        writer.release()

    print(f"Processed {frame_index} frames")
    print(f"Saved to: {output_path}")

    # export all player trajectories to JSON
    trajectory_path = (
        OUTPUT_DIR / f"{video_path.stem}_trajectories.json"
    )
    # for context, origin is far-left corner of doubles court
    # this matches how image coordinates work in OpenCV
    trajectory_data = {
        "video": video_path.name,
        "fps": fps,
        "coordinate_system": {
            "units": "meters",
            "origin": "far-left baseline corner",
            "x_direction": "toward far-right baseline corner",
            "y_direction": "toward near baseline"
        },
        "frames": trajectory_frames
    }
    # write the JSON file
    with open(trajectory_path, "w") as file:
        json.dump(trajectory_data, file, indent=2)
    print(f"Saved trajectories to: {trajectory_path}")

if __name__ == "__main__":
    main()
