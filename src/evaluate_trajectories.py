import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def position_error(predicted, ground_truth):
    # euclidean position error in meters
    if not predicted or not ground_truth:
        return None

    pred_pos = predicted.get("court_m")
    true_pos = ground_truth.get("court_m")

    if not pred_pos or not true_pos:
        return None

    return float(np.linalg.norm(np.array(pred_pos) - np.array(true_pos)))


def evaluate(annotations, raw_data, filtered_data):
    # index trajectory frames by their original frame numbers
    raw_frames = {f["frame"]: f for f in raw_data["frames"]}
    filtered_frames = {f["frame"]: f for f in filtered_data["frames"]}
    results = []

    for annotation in annotations["annotations"]:
        frame_idx = annotation["frame"]
        raw_frame = raw_frames.get(frame_idx)
        filtered_frame = filtered_frames.get(frame_idx)

        if not raw_frame or not filtered_frame:
            print(f"skipping missing frame {frame_idx}")
            continue

        for side in ("far", "near"):
            ground_truth = annotation["players"].get(side)
            if not ground_truth:
                continue

            raw_player = raw_frame["players"].get(side)
            filtered_player = filtered_frame["players"].get(side)

            source = filtered_player.get("position_source") if filtered_player else "missing"

            results.append({
                "frame": frame_idx,
                "timestamp_s": annotation["timestamp_s"],
                "player": side,
                "raw_error_m": position_error(raw_player, ground_truth),
                "filtered_error_m": position_error(filtered_player, ground_truth),
                "position_source": source
            })

    return results


def print_summary(results, label):
    rows = results if label == "overall" else [r for r in results if r["player"] == label]
    
    # paired eval where both methods have a position
    paired = [r for r in rows if r["raw_error_m"] is not None and r["filtered_error_m"] is not None]
    
    # missing raw measurements recovered by prediction
    predicted = [
        r for r in rows
        if r["raw_error_m"] is None 
        and r["filtered_error_m"] is not None 
        and r["position_source"] == "prediction"
    ]

    print(f"\n--- {label.upper()} ---")
    print(f"annotated positions: {len(rows)}")
    print(f"paired comparisons: {len(paired)}")

    print(f"raw missing: {sum(r['raw_error_m'] is None for r in rows)}")
    print(f"filtered missing: {sum(r['filtered_error_m'] is None for r in rows)}")

    if paired:
        raw_errors = np.array([r["raw_error_m"] for r in paired])
        filtered_errors = np.array([r["filtered_error_m"] for r in paired])

        raw_mean = np.mean(raw_errors)
        filt_mean = np.mean(filtered_errors)
        improvement = ((raw_mean - filt_mean) / raw_mean * 100) if raw_mean > 0 else 0

        print(f"raw mean err: {raw_mean:.3f} m")
        print(f"filt mean err: {filt_mean:.3f} m")
        print(f"raw median err: {np.median(raw_errors):.3f} m")
        print(f"filt median err: {np.median(filtered_errors):.3f} m")
        print(f"improvement: {improvement:+.2f}%")
        print(f"filtering better: {int(np.sum(filtered_errors < raw_errors))} times")
        print(f"filtering worse: {int(np.sum(filtered_errors > raw_errors))} times")

    if predicted:
        pred_errors = [r["filtered_error_m"] for r in predicted]
        print(f"recovered missing detections: {len(predicted)}")
        print(f"prediction mean err: {np.mean(pred_errors):.3f} m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--filtered", required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    raw_data = load_json(args.raw)
    filtered_data = load_json(args.filtered)

    # check all files refer to the same source video
    videos = {annotations["video"], raw_data["video"], filtered_data["video"]}
    if len(videos) != 1:
        raise ValueError("input files refer to different videos")

    results = evaluate(annotations, raw_data, filtered_data)
    if not results:
        raise RuntimeError("no matching annotated frames found")

    print(f"evaluating: {annotations['video']}")
    print_summary(results, "far")
    print_summary(results, "near")
    print_summary(results, "overall")

    # export per-frame errors for further analysis
    out_path = Path(args.filtered).parent / f"{Path(annotations['video']).stem}_evaluation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "timestamp_s", "player", "raw_error_m", 
                "filtered_error_m", "position_source"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nsaved eval csv: {out_path}")


if __name__ == "__main__":
    main()