from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import cv2
import io
from PIL import Image
from app.services.face_predict import predict_face_emotion

router = APIRouter()

@router.post("/predict-face")
async def predict_face(file: UploadFile = File(...)):
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image bytes and convert to OpenCV format
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Predict
    predicted_emotion = predict_face_emotion(img_bgr)

    return {"emotion": predicted_emotion}
