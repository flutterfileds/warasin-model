from fastapi import APIRouter
from pydantic import BaseModel
from app.services.journal_predict import predict_mood

router = APIRouter()

class JournalEntry(BaseModel):
    content: str

@router.post("/predict-journal")
def predict(entry: JournalEntry):
    mood = predict_mood(entry.content)
    return {"mood": mood}