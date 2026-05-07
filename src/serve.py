from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Bien moi truong se duoc cau hinh tren VM
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tai file model.pkl tu GCS ve may khi server khoi dong."""
    if not GCS_BUCKET:
        print("GCS_BUCKET environment variable not set. Skipping download.")
        return

    print(f"Downloading model from gs://{GCS_BUCKET}/{GCS_MODEL_KEY}...")
    
    # Dam bao thu muc ton tai
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model downloaded to {MODEL_PATH}")


# Goi ham nay khi server khoi dong
# Luu y: Trong moi truong thuc te, ham nay nen duoc goi boi server process
if os.environ.get("SKIP_MODEL_DOWNLOAD") != "true":
    try:
        download_model()
    except Exception as e:
        print(f"Error downloading model: {e}")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.
    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    Dau ra:  JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model exists on GCS.")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # Du doan tu model
    prediction = int(model.predict([req.features])[0])
    
    # Anh xa nhan
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    label = labels.get(prediction, "unknown")

    return {
        "prediction": prediction,
        "label": label
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
