import argparse
import json
from pathlib import Path
import numpy as np


class KalmanFilter:
    def __init__(self, fps, meas_std=0.4, acc_std=8.0, max_gap_s=0.5):
        self.dt = 1.0 / fps
        self.state = None
        self.P = None
        self.missed = 0
        self.max_missed = max(1, round(max_gap_s * fps))
        self.meas_std = meas_std

        # standard cv model matrices
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        self.R = np.eye(2) * meas_std**2
        G = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt]
        ])
        self.Q = acc_std**2 * (G @ G.T)

    def step(self, measurement):
        if self.state is None:
            if measurement is None:
                return None, "missing"
            
            # init on first observation
            self.state = np.array([measurement[0], measurement[1], 0, 0], dtype=float)
            self.P = np.diag([self.meas_std**2, self.meas_std**2, 9.0, 9.0])
            self.missed = 0
            return self.state[:2].copy(), "measurement"

        # predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        if measurement is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.state = None
                self.P = None
                return None, "missing"
            return self.state[:2].copy(), "prediction"

        self.missed = 0

        # update
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        self.state = self.state + K @ y
        I = np.eye(4)
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.state[:2].copy(), "measurement"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        data = json.load(f)
    fps = float(data["fps"])
    if fps <= 0:
        raise ValueError("fps must be > 0")
    filters = {"far": KalmanFilter(fps), "near": KalmanFilter(fps)}
    counts = {side: {"measurement": 0, "prediction": 0, "missing": 0} for side in filters}
    smoothed_frames = []
    for frame in data["frames"]:
        out_frame = {
            "frame": frame["frame"],
            "timestamp_s": frame["timestamp_s"],
            "players": {}
        }
        for side, raw_player in frame["players"].items():
            raw_pos = raw_player["court_m"] if raw_player else None
            pos, source = filters[side].step(raw_pos)
            counts[side][source] += 1
            if pos is None:
                out_frame["players"][side] = None
                continue
            # carry over existing metadata or init empty
            record = dict(raw_player) if raw_player else {"track_id": None, "foot_pixel": None}
            record.update({
                "raw_court_m": raw_pos,
                "court_m": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
                "position_source": source
            })
            out_frame["players"][side] = record
        smoothed_frames.append(out_frame)

    out_data = {
        **data,
        "filter": {"type": "constant_velocity_kalman"},
        "frames": smoothed_frames
    }

    out_path = in_path.with_name(f"{in_path.stem}_smoothed.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"saved: {out_path} ({len(smoothed_frames)} frames)")
    for side, stats in counts.items():
        print(f"{side}: {stats}")


if __name__ == "__main__":
    main()