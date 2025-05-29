from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import journal, face
from app.services.model_loader import load_models
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Load models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield


app = FastAPI(
    title="Warasin API",
    version="1.0.0",
    description="Emotion Detection API for Journal and Face",
    lifespan=lifespan,
)


@app.get("/")
def root():
    from app.services.model_loader import journal_model, face_model

    return {
        "message": "Warasin API is running",
        "version": "1.0.0",
        "models": {
            "journal_ready": journal_model is not None,
            "face_ready": face_model is not None,
        },
    }


@app.get("/health")
def health_check():
    from app.services.model_loader import journal_model, face_model

    return {
        "status": "healthy",
        "journal_model": "loaded" if journal_model else "not loaded",
        "face_model": "loaded" if face_model else "not loaded",
    }


app.include_router(
    journal.router, prefix="/api/v1", tags=["Journal Mood Classification"]
)
app.include_router(
    face.router, prefix="/api/v1", tags=["Facial Expression Recognition"]
)
