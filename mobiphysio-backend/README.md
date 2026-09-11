# MobiPhysio Backend

FastAPI backend for the MobiPhysio capstone MVP.

## What it does

1. Receives a user-uploaded physiotherapy video.
2. Samples 45 frames from the video.
3. Runs YOLOv8-Pose to extract 17 body keypoints.
4. Applies the same normalization idea used in training.
5. Builds the features expected by the saved models.
6. Predicts:
   - exercise type
   - AI-estimated quality score
7. Sends structured results to OpenAI for safe user feedback.
8. Returns JSON to the frontend.

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Render settings

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-luna
```

Do not commit the API key to GitHub.
