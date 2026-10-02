from deepface import DeepFace
from deepface.modules import exceptions
import cv2
import numpy as np
import traceback


def convert_image_to_cv2(image_file):
    """
    Converts an uploaded image file to OpenCV format
    """
    img_bytes = image_file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img


def validate_face(img):
    """
    Validates the face capture requirements like full, frontal, non-cropped face
    """
    
    try:
        # ---- FACE DETECTION ----
        detections = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True,
            anti_spoofing=False
        )
        
        # detect 1 face only
        if len(detections) != 1:
            return False, "Please ensure exactly one face is visible"

        face = detections[0]

        # Anti-spoofing -- NOT WORKING AT THE MOMENT
        if face.get("is_real") is False:
            return False, "Spoofing detected. Please use a live face."

        # Confidence check
        if face.get("confidence", 0) < 0.95:
            return False, "Face is not clearly visible"

        # FULL FACE VALIDATION - Reject partial face
        h_img, w_img, _ = img.shape
        area = face["facial_area"]
        
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        # 1️⃣ Face must NOT touch image borders (cropped face check)
        margin = 10  # pixels
        if (
            x <= margin or
            y <= margin or
            x + w >= w_img - margin or
            y + h >= h_img - margin
        ):
            return False, "Face is partially outside the frame"

        # 2️⃣ Face size must be sufficiently large
        face_area_ratio = (w * h) / (w_img * h_img)
        if face_area_ratio < 0.10:
            return False, "Face is too small or too far from camera"

        # 3️⃣ All key landmarks must exist
        landmarks = face.get("facial_area", {})

        required = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

        for key in required:
            if key not in landmarks:
                return False, "Face landmarks are incomplete"

        # 4️⃣ Eye alignment (reject tilted / partial faces)
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        eye_dx = abs(left_eye[0] - right_eye[0])
        eye_dy = abs(left_eye[1] - right_eye[1])

        if eye_dy / eye_dx > 0.15:
            return False, "Face is tilted or not frontal"
        
        emb = DeepFace.represent(img_path=img, model_name="Facenet512")[0]["embedding"]

    except exceptions.FaceNotDetected:
        return False, "No face detected in the image. Please try again."
    
    except Exception as e:
        traceback.print_exc()
        print(f"\033[91mError: {e}\033[0m")
        return False, "Internal server error during face processing. Contact Admin."

    
    print("Face validation passed ✅")
    return True, None


def generate_embedding(img):
    """
    Generates a facial embedding using Facenet512 model
    """
    try:
        emb = DeepFace.represent(img_path=img, model_name="Facenet512")[0]["embedding"]
        return emb
    except Exception as e:
        traceback.print_exc()
        print(f"\033[91mError generating embedding: {e}\033[0m")
        return None