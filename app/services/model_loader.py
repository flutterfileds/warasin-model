import joblib
import tensorflow as tf
import os
import logging
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

journal_model = None
vectorizer = None
encoder = None
face_model = None

async def load_models():
    """Load all models at startup"""
    global journal_model, vectorizer, encoder, face_model
    
    try:
        logger.info("Loading journal classification models...")
        journal_model = joblib.load("app/models/linear_svc_model.pkl")
        vectorizer = joblib.load("app/models/tfidf_vectorizer.pkl")
        encoder = joblib.load("app/models/label_encoder.pkl")
        logger.info("Journal models loaded successfully")
        
        await load_emotion_model_from_storage()
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

async def load_emotion_model_from_storage():
    """Load emotion recognition model from Azure Blob Storage"""
    global face_model
    
    try:
        model_path = "app/models/emotion_recognition_model.h5"
        
        if os.path.exists(model_path):
            logger.info("Loading emotion model from local cache")
        else:
            logger.info("Downloading emotion model from Azure Blob Storage")
            
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                logger.error("Azure Storage connection string not found")
                return
            
            os.makedirs("app/models", exist_ok=True)
            
            # Download model from blob storage
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(
                container="models", 
                blob="emotion_recognition_model.h5"
            )
            
            with open(model_path, "wb") as model_file:
                model_file.write(blob_client.download_blob().readall())
            
            logger.info("Emotion model downloaded successfully")
        
        # Load the Keras model
        face_model = tf.keras.models.load_model(model_path)
        logger.info("Emotion model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}")

def get_journal_models():
    """Get journal classification models"""
    return journal_model, vectorizer, encoder

def get_face_model():
    """Get facial emotion recognition model"""
    return face_model