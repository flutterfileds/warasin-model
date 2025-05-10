from fastapi import FastAPI
from app.routes import journal, face

app = FastAPI(title="Warasin API")

app.include_router(journal.router, prefix="/journal", tags=["Journal Mood Classification"])
app.include_router(face.router, prefix="/face", tags=["Facial Expression Recognition"])