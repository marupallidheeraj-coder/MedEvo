"""MedFusion-Evo backend (FastAPI).

Endpoints
  GET  /health         -> {"status": "ok", "trained": bool}
  POST /predict-image  -> multipart field "file" (JPG/PNG chest X-ray, max 8 MB)
                          returns {"p": pneumonia probability 0..1,
                                   "con": 64 Grad-CAM cell values (8x8 grid, row-major),
                                   "trained": bool, "model_version": str, "disclaimer": str}
Run:  uvicorn app:app --host 0.0.0.0 --port 8000

Configuration (environment variables, see ../.env.example)
  MODEL_PATH          path to model.pt (default: ./model.pt; untrained weights if missing)
  ALLOWED_ORIGINS     comma-separated CORS origins (default: local development only)
  RATE_LIMIT_PER_MIN  max /predict-image calls per client IP per minute (default 30)

Privacy: uploaded images are decoded in memory and never written to disk or logged.
"""
import logging
import os

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from heatmap import cam_to_grid
from model import load_model, predict
from security import MAX_BYTES, ImageRejected, RateLimiter, parse_origins, validate_image_bytes

log = logging.getLogger("medfusion")

WEIGHTS = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.pt"))
DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500,null"
ORIGINS = parse_origins(os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS))
MODEL_VERSION = "resnet18-imagenet-finetune-v1"
DISCLAIMER = "AI-assisted screening aid, not a medical diagnosis."

app = FastAPI(title="MedFusion-Evo API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
limiter = RateLimiter(limit=int(os.environ.get("RATE_LIMIT_PER_MIN", "30")), window=60.0)
model, trained = load_model(WEIGHTS)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"          # never cache medical images or results
    return response


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(429, "Too many requests, try again in a minute", headers={"Retry-After": "60"})


@app.get("/health")
def health():
    return {"status": "ok", "trained": trained}


# `async def` on purpose: requests are handled one at a time, which keeps the Grad-CAM
# forward hooks on the shared model from interleaving. Scale out with more workers instead.
@app.post("/predict-image", dependencies=[Depends(rate_limit)])
async def predict_image(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    try:
        img = validate_image_bytes(raw)
    except ImageRejected as e:
        raise HTTPException(e.status, e.detail) from None
    try:
        p, cam = predict(model, img)
        return {
            "p": round(p, 4),
            "con": cam_to_grid(cam, p),
            "trained": trained,
            "model_version": MODEL_VERSION,
            "disclaimer": DISCLAIMER,
        }
    except Exception:                                        # never leak internals to the client
        log.exception("prediction failed")
        raise HTTPException(500, "Prediction failed") from None
