from app.utils.text_cleaning import clean_text

def predict_mood(text, journal_model, vectorizer, encoder):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    predicted_label = journal_model.predict(text_tfidf)[0]
    emotion = encoder.inverse_transform([predicted_label])[0]
    return emotion