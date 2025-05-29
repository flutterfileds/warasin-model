from app.utils.text_cleaning import clean_text
from app.services.model_loader import get_journal_models

def predict_mood(text):
    model, vectorizer, encoder = get_journal_models()
    
    if model is None or vectorizer is None or encoder is None:
        raise Exception("Journal models not loaded")
    
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    predicted_label = model.predict(text_tfidf)[0]
    emotion = encoder.inverse_transform([predicted_label])[0]
    return emotion