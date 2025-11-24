from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from src.models.predict import predict_from_url


class PredictRequest(BaseModel):
    image_url: HttpUrl


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]


app = FastAPI(
    title="Hotdog or Legs API",
    description="Classify an image URL as 'hotdog' or 'legs'.",
    version="0.1.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = predict_from_url(str(req.image_url))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return PredictResponse(
        label=result["pred_label"],
        probabilities=result["probs"],
    )
