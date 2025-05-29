import joblib
import tensorflow as tf
import os
import logging
import asyncio
from azure.storage.blob import BlobServiceClient
from typing import Optional, Tuple
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
journal_model = None
vectorizer = None
encoder = None
face_model = None

# Model loading status
models_loaded = {"journal": False, "face": False}


async def load_models():
    """Load all models at startup"""
    global journal_model, vectorizer, encoder, face_model, models_loaded

    logger.info("Starting model loading process...")

    try:
        # Load journal models first (faster to load)
        await load_journal_models()

        # Load face emotion model (slower, from Azure)
        await load_emotion_model_from_storage()

        logger.info("All models loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e


async def load_journal_models():
    """Load journal classification models"""
    global journal_model, vectorizer, encoder, models_loaded

    try:
        logger.info("Loading journal classification models...")

        # Check if models exist locally
        model_files = {
            "model": "app/models/linear_svc_model.pkl",
            "vectorizer": "app/models/tfidf_vectorizer.pkl",
            "encoder": "app/models/label_encoder.pkl",
        }

        # Verify all files exist
        missing_files = []
        for name, path in model_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")

        if missing_files:
            logger.error(f"Missing journal model files: {missing_files}")
            raise FileNotFoundError(f"Missing journal model files: {missing_files}")

        # Load models
        journal_model = joblib.load(model_files["model"])
        vectorizer = joblib.load(model_files["vectorizer"])
        encoder = joblib.load(model_files["encoder"])

        # Verify models are loaded correctly
        if journal_model is None or vectorizer is None or encoder is None:
            raise ValueError("One or more journal models failed to load properly")

        models_loaded["journal"] = True
        logger.info("Journal models loaded successfully")

        # Log model information
        logger.info(f"Journal model type: {type(journal_model).__name__}")
        logger.info(f"Vectorizer type: {type(vectorizer).__name__}")
        logger.info(
            f"Encoder classes: {encoder.classes_ if hasattr(encoder, 'classes_') else 'Unknown'}"
        )

    except Exception as e:
        logger.error(f"Failed to load journal models: {e}")
        models_loaded["journal"] = False
        raise e


async def load_emotion_model_from_storage():
    """Load emotion recognition model from Azure Blob Storage or local cache"""
    global face_model, models_loaded

    try:
        model_path = "app/models/emotion_recognition_model.h5"

        # Create models directory if it doesn't exist
        os.makedirs("app/models", exist_ok=True)

        # Check if model exists locally
        if os.path.exists(model_path):
            logger.info("Loading emotion model from local cache")
            face_model = tf.keras.models.load_model(model_path)
        else:
            logger.info("Model not found locally, downloading from Azure Blob Storage")

            # Get Azure connection string
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                logger.error(
                    "Azure Storage connection string not found in environment variables"
                )
                raise ValueError(
                    "AZURE_STORAGE_CONNECTION_STRING environment variable is required"
                )

            # Download model from Azure Blob Storage
            await download_model_from_azure(connection_string, model_path)

            # Load the downloaded model
            face_model = tf.keras.models.load_model(model_path)

        # Verify model is loaded correctly
        if face_model is None:
            raise ValueError("Face emotion model failed to load properly")

        models_loaded["face"] = True
        logger.info("Emotion model loaded successfully")

        # Log model information
        logger.info(f"Face model input shape: {face_model.input_shape}")
        logger.info(f"Face model output shape: {face_model.output_shape}")

    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}")
        models_loaded["face"] = False
        # Don't raise exception here to allow journal model to work independently
        logger.warning("Face emotion recognition will not be available")


async def download_model_from_azure(connection_string: str, local_path: str):
    """Download model from Azure Blob Storage"""
    try:
        logger.info("Connecting to Azure Blob Storage...")

        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get blob client
        container_name = "models"
        blob_name = "emotion_recognition_model.h5"

        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        # Check if blob exists
        if not blob_client.exists():
            raise FileNotFoundError(
                f"Model blob not found: {container_name}/{blob_name}"
            )

        logger.info(f"Downloading model from {container_name}/{blob_name}...")

        # Download blob data
        blob_data = blob_client.download_blob().readall()

        # Save to local file
        with open(local_path, "wb") as model_file:
            model_file.write(blob_data)

        logger.info(f"Model downloaded successfully to {local_path}")

        # Verify file was written correctly
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise ValueError("Downloaded model file is empty or corrupted")

        logger.info(f"Model file size: {os.path.getsize(local_path)} bytes")

    except Exception as e:
        logger.error(f"Failed to download model from Azure: {e}")
        raise e


def get_journal_models() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Get journal classification models"""
    return journal_model, vectorizer, encoder


def get_face_model() -> Optional[object]:
    """Get facial emotion recognition model"""
    return face_model


def is_journal_model_ready() -> bool:
    """Check if journal models are ready"""
    return models_loaded["journal"] and all(
        [journal_model is not None, vectorizer is not None, encoder is not None]
    )


def is_face_model_ready() -> bool:
    """Check if face model is ready"""
    return models_loaded["face"] and face_model is not None


def get_model_status() -> dict:
    """Get status of all models"""
    return {
        "journal": {
            "loaded": is_journal_model_ready(),
            "model_type": type(journal_model).__name__ if journal_model else None,
            "vectorizer_type": type(vectorizer).__name__ if vectorizer else None,
            "encoder_classes": (
                encoder.classes_.tolist()
                if encoder and hasattr(encoder, "classes_")
                else None
            ),
        },
        "face": {
            "loaded": is_face_model_ready(),
            "model_type": type(face_model).__name__ if face_model else None,
            "input_shape": face_model.input_shape if face_model else None,
            "output_shape": face_model.output_shape if face_model else None,
        },
    }


async def reload_models():
    """Reload all models (useful for updates)"""
    global journal_model, vectorizer, encoder, face_model, models_loaded

    logger.info("Reloading all models...")

    # Reset global variables
    journal_model = None
    vectorizer = None
    encoder = None
    face_model = None
    models_loaded = {"journal": False, "face": False}

    # Reload models
    await load_models()


def cleanup_models():
    """Cleanup loaded models to free memory"""
    global journal_model, vectorizer, encoder, face_model, models_loaded

    logger.info("Cleaning up models...")

    journal_model = None
    vectorizer = None
    encoder = None
    face_model = None
    models_loaded = {"journal": False, "face": False}

    # Force garbage collection
    import gc

    gc.collect()

    logger.info("Models cleaned up successfully")


# Model validation functions
def validate_journal_models():
    """Validate journal models are working correctly"""
    try:
        if not is_journal_model_ready():
            return False, "Journal models not loaded"

        # Test with sample data
        test_text = "I am feeling happy today"
        from app.utils.text_cleaning import clean_text

        cleaned = clean_text(test_text)
        text_tfidf = vectorizer.transform([cleaned])
        predicted_label = journal_model.predict(text_tfidf)[0]
        emotion = encoder.inverse_transform([predicted_label])[0]

        logger.info(
            f"Journal model validation successful: '{test_text}' -> '{emotion}'"
        )
        return True, f"Working correctly, predicted: {emotion}"

    except Exception as e:
        logger.error(f"Journal model validation failed: {e}")
        return False, str(e)


def validate_face_model():
    """Validate face model is working correctly"""
    try:
        if not is_face_model_ready():
            return False, "Face model not loaded"

        # Test with dummy data
        import numpy as np

        dummy_input = np.random.rand(1, 224, 224, 3)
        predictions = face_model.predict(dummy_input, verbose=0)

        if predictions is None or len(predictions) == 0:
            return False, "Model returned empty predictions"

        logger.info("Face model validation successful")
        return True, f"Working correctly, output shape: {predictions.shape}"

    except Exception as e:
        logger.error(f"Face model validation failed: {e}")
        return False, str(e)


# Health check function
def health_check() -> dict:
    """Comprehensive health check for all models"""
    journal_valid, journal_msg = validate_journal_models()
    face_valid, face_msg = validate_face_model()

    return {
        "overall_status": (
            "healthy"
            if journal_valid and face_valid
            else "partial" if journal_valid or face_valid else "unhealthy"
        ),
        "journal": {
            "status": "healthy" if journal_valid else "unhealthy",
            "message": journal_msg,
        },
        "face": {
            "status": "healthy" if face_valid else "unhealthy",
            "message": face_msg,
        },
        "models_loaded": models_loaded,
    }
