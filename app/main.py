from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="Toxic Comment Classifier API")

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class CommentRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Toxic Comment Classifier API is running"}

@app.post("/predict")
def predict(request: CommentRequest):
    probs = model.predict_proba([request.text])[0]
    preds = model.predict([request.text])[0]

    result = {}
    for label, prob, pred in zip(LABELS, probs, preds):
        result[label] = {
            "predicted": bool(pred),
            "probability": float(prob)
        }

    return {
        "text": request.text,
        "predictions": result
    }