from fastapi import APIRouter
from pydantic import BaseModel
from app.services.journal_predict import predict_mood
from app.services.model_loader import journal_model, vectorizer, encoder

router = APIRouter()

class JournalEntry(BaseModel):
    text: str

@router.post("/predict-journal")
def predict(entry: JournalEntry):
    mood = predict_mood(entry.text, journal_model, vectorizer, encoder)
    return {"mood": mood}