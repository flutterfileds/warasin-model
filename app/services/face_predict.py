import numpy as np
import cv2
from app.services.model_loader import face_model
import tensorflow as tf

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_face_emotion(image_array: np.ndarray) -> str:
    rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_img, 1.32, 5)

    if len(faces) == 0:
        return "no face detected"

    x, y, w, h = faces[0]
    roi = rgb_img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (224, 224))

    img_pixels = tf.keras.utils.img_to_array(roi)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0

    predictions = face_model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    return emotions[max_index]
