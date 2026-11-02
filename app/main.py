from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = pipeline("text-classification", model=MODEL_NAME)
    yield


app = FastAPI(title="Sentiment API", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    result = classifier(request.text)[0]
    return PredictionResponse(label=result["label"], score=result["score"])