"""Pose estimation: video file -> cached landmark sequence (.npz).

Uses the MediaPipe Tasks PoseLandmarker (33 landmarks). Frames are sampled at
`config.TARGET_FPS`, downscaled, and run in VIDEO mode so tracking is used between
frames. Frames without a detected pose are stored as NaN with detected=False so the
downstream code can decide whether a clip is analyzable (proposal, section 15:
explicit unable-to-analyze state rather than fabricated results).

Cached file layout (np.savez_compressed):
    landmarks  (T, 33, 4) float32  normalized image coords x, y in [0,1], z, visibility
    world      (T, 33, 3) float32  metric world landmarks (hip-centred, metres)
    detected   (T,)       bool
    timestamps (T,)       float32  seconds
    meta       str (JSON): video_id, fps, width, height, n_frames, sampled_fps, duration_s
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physio import config

N_LANDMARKS = 33
_LANDMARKER = None  # per-process singleton


@dataclass
class PoseSequence:
    landmarks: np.ndarray            # (T, 33, 4)
    world: np.ndarray                # (T, 33, 3)
    detected: np.ndarray             # (T,)
    timestamps: np.ndarray           # (T,)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if self.n_frames else 0.0

    @property
    def n_detected(self) -> int:
        return int(self.detected.sum())

    @property
    def mean_visibility(self) -> float:
        vis = self.landmarks[self.detected][:, :, 3]
        return float(np.nanmean(vis)) if vis.size else 0.0

    def quality_gate(self) -> tuple[bool, str]:
        """Return (ok, reason). Mirrors the thresholds in config."""
        if self.n_detected < config.MIN_VALID_FRAMES:
            return False, f"only {self.n_detected} frames with a detected pose (min {config.MIN_VALID_FRAMES})"
        if self.detection_rate < config.MIN_DETECTION_RATE:
            return False, f"pose detected in {self.detection_rate:.0%} of frames (min {config.MIN_DETECTION_RATE:.0%})"
        if self.mean_visibility < config.MIN_MEAN_VISIBILITY:
            return False, f"mean landmark visibility {self.mean_visibility:.2f} (min {config.MIN_MEAN_VISIBILITY})"
        return True, "ok"

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, landmarks=self.landmarks.astype(np.float32),
                            world=self.world.astype(np.float32), detected=self.detected.astype(bool),
                            timestamps=self.timestamps.astype(np.float32), meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as z:
            return cls(landmarks=z["landmarks"], world=z["world"], detected=z["detected"],
                       timestamps=z["timestamps"], meta=json.loads(str(z["meta"])))


def _get_landmarker(model_path: Path = config.POSE_MODEL_PATH):
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not Path(model_path).exists():
            from physio.data.download import download_pose_model
            download_pose_model(model_path)
        # CPU delegate explicitly: on macOS the default may select the Metal GPU delegate,
        # which aborts the process ("DrishtiMetalHelper ... Service is unavailable") when
        # running inside a server worker thread.
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path),
                                               delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _LANDMARKER = vision.PoseLandmarker.create_from_options(options)
    return _LANDMARKER


def reset_landmarker() -> None:
    """VIDEO mode requires monotonically increasing timestamps per stream; recreate between videos."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        try:
            _LANDMARKER.close()
        except Exception:  # noqa: BLE001
            pass
    _LANDMARKER = None


def iter_sampled_frames(video_path: Path, target_fps: float = config.TARGET_FPS,
                        max_side: int = config.MAX_FRAME_SIDE):
    """Yield (frame_index, timestamp_s, rgb_frame) at approximately target_fps."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / target_fps)))
    info = {"fps": float(fps), "n_frames": n_frames, "step": step,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield idx, idx / fps, rgb, info
            idx += 1
    finally:
        cap.release()


def extract_pose_sequence(video_path: Path, video_id: str | None = None,
                          target_fps: float = config.TARGET_FPS) -> PoseSequence:
    import mediapipe as mp
    reset_landmarker()
    landmarker = _get_landmarker()
    lms, worlds, det, ts = [], [], [], []
    info: dict = {}
    last_ts_ms = -1
    for idx, t, rgb, info in iter_sampled_frames(video_path, target_fps):
        ts_ms = int(round(t * 1000))
        if ts_ms <= last_ts_ms:            # MediaPipe VIDEO mode requires strictly increasing timestamps
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = landmarker.detect_for_video(image, ts_ms)
        if res.pose_landmarks:
            p = res.pose_landmarks[0]
            w = res.pose_world_landmarks[0]
            lms.append([[l.x, l.y, l.z, (l.visibility if l.visibility is not None else 0.0)] for l in p])
            worlds.append([[l.x, l.y, l.z] for l in w])
            det.append(True)
        else:
            lms.append(np.full((N_LANDMARKS, 4), np.nan))
            worlds.append(np.full((N_LANDMARKS, 3), np.nan))
            det.append(False)
        ts.append(t)
    reset_landmarker()
    if not ts:
        raise IOError(f"no frames decoded from {video_path}")
    meta = {"video_id": video_id or Path(video_path).stem, "fps": info.get("fps"), "width": info.get("width"),
            "height": info.get("height"), "n_frames": info.get("n_frames"),
            "sampled_fps": info.get("fps", 30.0) / info.get("step", 1), "duration_s": float(ts[-1]),
            "target_fps": target_fps, "n_sampled": len(ts)}
    return PoseSequence(landmarks=np.asarray(lms, dtype=np.float32), world=np.asarray(worlds, dtype=np.float32),
                        detected=np.asarray(det, dtype=bool), timestamps=np.asarray(ts, dtype=np.float32), meta=meta)


def pose_cache_path(video_id: str, poses_dir: Path | None = None) -> Path:
    return Path(poses_dir if poses_dir is not None else config.POSES_DIR) / f"{video_id}.npz"
