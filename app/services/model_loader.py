import joblib
import tensorflow as tf

journal_model = joblib.load("app/models/linear_svc_model.pkl")
vectorizer = joblib.load("app/models/tfidf_vectorizer.pkl")
encoder = joblib.load("app/models/label_encoder.pkl")
face_model = tf.keras.models.load_model("app/models/emotion_recognition_model.h5")