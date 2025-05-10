from fastapi import APIRouter
from pydantic import BaseModel
from app.services.journal_predict import predict_mood

router = APIRouter()

class JournalEntry(BaseModel):
    text: str

@router.post("/predict")
def predict(entry: JournalEntry):
    mood = predict_mood(entry.text)
    return {"mood": mood}