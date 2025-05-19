from fastapi import FastAPI
from app.routes import journal, face

app = FastAPI(title="Warasin API")

app.include_router(journal.router, tags=["Journal Mood Classification"])
app.include_router(face.router, tags=["Facial Expression Recognition"])