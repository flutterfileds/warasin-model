import joblib

model = joblib.load("app/models/linear_svc_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")
encoder = joblib.load("app/models/encoder.pkl")