import os
import json
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# YOLO is loaded lazily, so /health can work even before the first prediction.
_yolo_model = None

APP_TITLE = "MobiPhysio Backend API"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "final_classification_model.joblib")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "final_regression_model.joblib")

# Exercise mapping used during training.
IDX_TO_EXERCISE = {
    0: "E01 - Abduction",
    1: "E02 - Adduction",
    2: "E03 - Lateral Rotation",
    3: "E04 - Medial Rotation",
    4: "E05 - Circumduction",
    5: "E06 - Wrist Extension",
    6: "E07 - Hip Joint Flexion",
    7: "E08 - Lumbar Flexion",
    8: "E09 - Back Extension",
}

# COCO-17 keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_CONF_FOR_REFERENCE = 0.10

ANGLE_TRIPLETS = {
    "left_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "left_shoulder": (L_ELBOW, L_SHOULDER, L_HIP),
    "right_shoulder": (R_ELBOW, R_SHOULDER, R_HIP),
    "left_hip": (L_SHOULDER, L_HIP, L_KNEE),
    "right_hip": (R_SHOULDER, R_HIP, R_KNEE),
    "left_knee": (L_HIP, L_KNEE, L_ANKLE),
    "right_knee": (R_HIP, R_KNEE, R_ANKLE),
}

SYMMETRY_PAIRS = [
    ("left_elbow", "right_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
]

DISCLAIMER = (
    "This feedback is for exercise-support purposes only and is not a medical diagnosis "
    "or a replacement for guidance from a qualified physiotherapist."
)

app = FastAPI(title=APP_TITLE)

# Allow Emergent/Lovable/any frontend during MVP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classification_model = joblib.load(CLASS_MODEL_PATH)
regression_model = joblib.load(REG_MODEL_PATH)


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n-pose.pt")
    return _yolo_model


def sample_45_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Uploaded video has no readable frames.")

    sampled_indices = np.linspace(0, total_frames - 1, 45).astype(int)
    frames = []

    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        else:
            frames.append(None)

    cap.release()
    return frames


def extract_pose_from_video(video_path: str) -> np.ndarray:
    """Return raw YOLO pose array with shape (45, 17, 3)."""
    model = get_yolo_model()
    frames = sample_45_frames(video_path)

    output = []
    for frame in frames:
        kp = np.zeros((17, 3), dtype=np.float32)

        if frame is not None:
            result = model.predict(frame, verbose=False)[0]

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                xy = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()
                kp = np.c_[xy, conf].astype(np.float32)

        output.append(kp)

    return np.asarray(output, dtype=np.float32)


def _valid_point(frame: np.ndarray, idx: int) -> Optional[np.ndarray]:
    point = frame[idx]
    if point[2] >= MIN_CONF_FOR_REFERENCE and np.isfinite(point[:2]).all():
        return point[:2].astype(np.float32)
    return None


def _hip_center(frame: np.ndarray) -> Optional[np.ndarray]:
    left = _valid_point(frame, L_HIP)
    right = _valid_point(frame, R_HIP)

    if left is not None and right is not None:
        return (left + right) / 2.0
    if left is not None:
        return left
    if right is not None:
        return right
    return None


def _torso_length(frame: np.ndarray) -> Optional[float]:
    left_sh = _valid_point(frame, L_SHOULDER)
    right_sh = _valid_point(frame, R_SHOULDER)
    left_hip = _valid_point(frame, L_HIP)
    right_hip = _valid_point(frame, R_HIP)

    if left_sh is None or right_sh is None or left_hip is None or right_hip is None:
        return None

    shoulder_mid = (left_sh + right_sh) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    length = float(np.linalg.norm(shoulder_mid - hip_mid))

    if not np.isfinite(length) or length <= 1e-6:
        return None

    return length


def normalize_pose_sequence(raw_pose: np.ndarray) -> np.ndarray:
    """Normalize a single raw pose sequence to shape (45, 17, 3)."""
    raw_pose = np.asarray(raw_pose, dtype=np.float32)
    if raw_pose.shape != (45, 17, 3):
        raise ValueError(f"Expected pose shape (45, 17, 3), got {raw_pose.shape}")

    scales = []
    for frame in raw_pose:
        length = _torso_length(frame)
        if length is not None:
            scales.append(length)

    scale = float(np.median(scales)) if scales else 1.0
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0

    normalized = raw_pose.copy()

    for i, frame in enumerate(raw_pose):
        center = _hip_center(frame)

        if center is None:
            # Keep confidence, but zero spatial coordinates for invalid reference frames.
            normalized[i, :, :2] = 0.0
            normalized[i, :, 2] = frame[:, 2]
        else:
            normalized[i, :, :2] = (frame[:, :2] - center) / scale
            normalized[i, :, 2] = frame[:, 2]

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32)


def flatten_pose(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float32)


def extract_engineered_features(X: np.ndarray) -> np.ndarray:
    features = []

    for sample in X:
        xy = sample[:, :, :2]
        conf = sample[:, :, 2]

        sample_features = []
        sample_features.extend(np.mean(xy, axis=0).flatten())
        sample_features.extend(np.std(xy, axis=0).flatten())
        sample_features.extend(np.min(xy, axis=0).flatten())
        sample_features.extend(np.max(xy, axis=0).flatten())

        movement = np.diff(xy, axis=0)
        movement_mag = np.sqrt(np.sum(movement ** 2, axis=-1))
        sample_features.extend(np.mean(movement_mag, axis=0).flatten())

        sample_features.extend(np.mean(conf, axis=0).flatten())
        features.append(sample_features)

    return np.asarray(features, dtype=np.float32)


def _safe_stats(x: np.ndarray) -> List[float]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [0.0] * 8
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.percentile(x, 25)),
        float(np.median(x)),
        float(np.percentile(x, 75)),
        float(np.max(x)),
        float(np.max(x) - np.min(x)),
    ]


def _angle_series(xy: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    ba = xy[:, a] - xy[:, b]
    bc = xy[:, c] - xy[:, b]

    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    denom = np.maximum(ba_norm * bc_norm, 1e-8)

    cosine = np.sum(ba * bc, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def extract_kinematic_quality_features(X: np.ndarray) -> np.ndarray:
    output = []

    for sample in X:
        xy = sample[:, :, :2].astype(np.float32)
        conf = sample[:, :, 2].astype(np.float32)

        f = []

        for kp in range(xy.shape[1]):
            f.extend(_safe_stats(xy[:, kp, 0]))
            f.extend(_safe_stats(xy[:, kp, 1]))

        velocity = np.diff(xy, axis=0)
        speed = np.linalg.norm(velocity, axis=-1)

        acceleration = np.diff(velocity, axis=0)
        accel_mag = np.linalg.norm(acceleration, axis=-1)

        for kp in range(xy.shape[1]):
            f.extend(_safe_stats(speed[:, kp]))
            f.extend(_safe_stats(accel_mag[:, kp]))

        angle_cache = {}
        for name, (a, b, c) in ANGLE_TRIPLETS.items():
            angles = _angle_series(xy, a, b, c)
            angle_cache[name] = angles

            f.extend(_safe_stats(angles))
            angular_velocity = np.diff(angles)
            f.extend(_safe_stats(np.abs(angular_velocity)))

        for left_name, right_name in SYMMETRY_PAIRS:
            symmetry_error = np.abs(angle_cache[left_name] - angle_cache[right_name])
            f.extend(_safe_stats(symmetry_error))

        shoulder_mid = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0
        hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
        torso_vec = shoulder_mid - hip_mid

        torso_angle = np.degrees(np.arctan2(torso_vec[:, 0], -torso_vec[:, 1] + 1e-8))
        f.extend(_safe_stats(torso_angle))

        hip_motion = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1)
        shoulder_motion = np.linalg.norm(np.diff(shoulder_mid, axis=0), axis=1)
        f.extend(_safe_stats(hip_motion))
        f.extend(_safe_stats(shoulder_motion))

        global_speed = np.mean(speed, axis=1)
        global_accel = np.mean(accel_mag, axis=1)
        f.extend(_safe_stats(global_speed))
        f.extend(_safe_stats(global_accel))

        f.extend(_safe_stats(conf.flatten()))
        f.extend(np.mean(conf, axis=0).tolist())

        output.append(f)

    return np.asarray(output, dtype=np.float32)


def exercise_one_hot(labels: np.ndarray, n_classes: int = 9) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def build_quality_feature_matrix(X: np.ndarray, X_basic_eng: np.ndarray, exercise_labels: np.ndarray) -> np.ndarray:
    kin = extract_kinematic_quality_features(X)
    ex = exercise_one_hot(exercise_labels, 9)
    return np.hstack([X_basic_eng, kin, ex]).astype(np.float32)


def deterministic_feedback(payload: Dict[str, Any]) -> str:
    exercise = payload["exercise"]
    score = payload["ai_estimated_quality_score"]
    confidence = payload["classification_confidence"]

    parts = [
        f"Exercise detected: {exercise} (confidence {confidence:.0%}).",
        f"AI-estimated movement quality: {score:.1f}/100.",
    ]

    observations = payload.get("observations") or []
    if observations:
        parts.append("Movement observations: " + "; ".join(observations[:3]) + ".")

    parts.append(DISCLAIMER)
    return " ".join(parts)


def build_llm_payload(
    exercise: str,
    classification_confidence: float,
    ai_estimated_quality_score: float,
    observations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "exercise": str(exercise),
        "classification_confidence": float(np.clip(classification_confidence, 0.0, 1.0)),
        "ai_estimated_quality_score": float(np.clip(ai_estimated_quality_score, 0.0, 100.0)),
        "observations": list(observations or []),
    }


def feedback_is_safe(text: str) -> bool:
    unsafe_patterns = [
        r"\byou have (?:an? )?[a-z]",
        r"\byou suffer from\b",
        r"\byour diagnosis is\b",
        r"\byou are diagnosed with\b",
        r"\bthis indicates (?:an? )?(?:injury|disease|disorder)\b",
        r"\byou need (?:medical )?treatment\b",
        r"\bi prescribe\b",
        r"\byou should take (?:medication|medicine|drugs?)\b",
    ]
    lowered = text.lower()
    return not any(__import__("re").search(pattern, lowered) for pattern in unsafe_patterns)


def ensure_safe_feedback(text: str, payload: Dict[str, Any]) -> str:
    if not text or not feedback_is_safe(text):
        return deterministic_feedback(payload)

    if "not a medical diagnosis" not in text.lower():
        text = text.rstrip() + " " + DISCLAIMER

    return text


def llm_feedback(payload: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    if not api_key:
        return deterministic_feedback(payload)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = """
You are the feedback-writing layer of MobiPhysio,
a home physiotherapy exercise-support prototype.

Use only the validated structured values supplied by the application.

Rules:
- Do not diagnose injuries, diseases, or medical conditions.
- Do not invent exercise labels, measurements, repetitions, or scores.
- Call the score an AI-estimated movement-quality score.
- Only mention observations supplied in the structured input.
- Do not generate new exercise instructions or corrective actions unless they are explicitly provided in the structured input.
- Keep the response concise, supportive, and easy to understand.
- Always state that the feedback is not a medical diagnosis and does not replace guidance from a qualified physiotherapist.
""".strip()

        user_prompt = (
            "Generate concise user feedback from this validated application result:\n"
            + json.dumps(payload, indent=2)
        )

        response = client.responses.create(
            model=model_name,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=300,
        )
        return ensure_safe_feedback(response.output_text.strip(), payload)

    except Exception:
        # Safe fallback keeps demo running if OpenAI fails.
        return deterministic_feedback(payload)


def predict_from_pose(normalized_pose: np.ndarray) -> Dict[str, Any]:
    X = normalized_pose[None, :, :, :].astype(np.float32)

    X_eng = extract_engineered_features(X)
    cls_pred = int(classification_model.predict(X_eng)[0])

    if hasattr(classification_model, "predict_proba"):
        proba = classification_model.predict_proba(X_eng)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 0.0

    exercise_name = IDX_TO_EXERCISE.get(cls_pred, str(cls_pred))

    # The regression model expects 948 features.
    # In this project that is: 170 engineered features + kinematic features + 9 one-hot exercise labels.
    if getattr(regression_model, "n_features_in_", None) == 948:
        X_reg = build_quality_feature_matrix(X, X_eng, np.array([cls_pred], dtype=np.int64))
    elif getattr(regression_model, "n_features_in_", None) == 2295:
        X_reg = flatten_pose(X)
    else:
        # Fallback: choose by feature size.
        raw = flatten_pose(X)
        X_reg = raw if raw.shape[1] == getattr(regression_model, "n_features_in_", -1) else X_eng

    quality_score = float(regression_model.predict(X_reg)[0])
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    observations = []
    if confidence < 0.60:
        observations.append("classification confidence is low")
    if quality_score < 60:
        observations.append("the estimated quality score is low")

    payload = build_llm_payload(
        exercise=exercise_name,
        classification_confidence=confidence,
        ai_estimated_quality_score=quality_score,
        observations=observations,
    )
    feedback = llm_feedback(payload)

    return {
        "exercise": exercise_name,
        "exercise_index": cls_pred,
        "confidence": confidence,
        "quality_score": quality_score,
        "feedback": feedback,
        "payload": payload,
    }


@app.get("/")
def root():
    return {"message": "MobiPhysio backend is running. Open /docs to test the API."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classification_model_features": int(getattr(classification_model, "n_features_in_", -1)),
        "regression_model_features": int(getattr(regression_model, "n_features_in_", -1)),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = (".mp4", ".mov", ".avi", ".mkv")
    filename = file.filename or "uploaded_video.mp4"

    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file such as MP4, MOV, AVI, or MKV."
        )

    suffix = os.path.splitext(filename)[1] or ".mp4"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        raw_pose = extract_pose_from_video(temp_path)
        normalized_pose = normalize_pose_sequence(raw_pose)
        result = predict_from_pose(normalized_pose)

        return {
            "success": True,
            "filename": filename,
            **result,
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
