from app.services.model_loader import model, vectorizer, encoder

def predict_mood(text: str) -> str:
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    emotion = encoder.inverse_transform(prediction)[0]  # Convert to readable emotion
    return emotion
