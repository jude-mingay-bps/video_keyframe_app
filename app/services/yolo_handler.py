import os
import glob
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from config import Config

# Global variable to store loaded model
loaded_model = None
model_info = None


def load_yolo_model():
    """Load the first available YOLO model from the models folder."""
    global loaded_model, model_info

    if loaded_model is not None:
        return loaded_model, model_info

    model_extensions = ['*.pt', '*.onnx', '*.engine']
    model_files = []

    for ext in model_extensions:
        model_files.extend(glob.glob(os.path.join(Config.MODELS_FOLDER, ext)))

    if not model_files:
        print("No YOLO model found in models folder")
        return None, None

    model_path = model_files[0]

    try:
        print(f"Loading YOLO model from: {model_path}")
        loaded_model = YOLO(model_path)

        model_info = {
            'path': model_path,
            'name': os.path.basename(model_path),
            'classes': loaded_model.names if hasattr(loaded_model, 'names') else {},
            'num_classes': len(loaded_model.names) if hasattr(loaded_model, 'names') else 0
        }

        print(f"Model loaded successfully: {model_info['name']}")
        print(f"Classes: {list(model_info['classes'].values())}")

        return loaded_model, model_info

    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None


def predict_on_frame(frame_base64, confidence_threshold=0.25):
    """Run YOLO prediction on a frame."""
    model, info = load_yolo_model()

    if model is None:
        return None, "No YOLO model available"

    try:
        image_bytes = base64.b64decode(frame_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return None, "Failed to decode image"

        results = model(frame, conf=confidence_threshold)

        if not results or len(results) == 0:
            return [], "No detections"

        result = results[0]

        annotations = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                img_height, img_width = frame.shape[:2]
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                class_name = info['classes'].get(class_id, f'class_{class_id}')

                annotations.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                    'bbox_yolo': [float(x_center), float(y_center), float(width), float(height)]
                })

        return annotations, "Success"

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, f"Prediction error: {str(e)}"


def create_yolo_annotation_file(annotations, image_width, image_height):
    """Create YOLO format annotation text."""
    lines = []
    for ann in annotations:
        class_id = ann['class_id']
        x_center, y_center, width, height = ann['bbox_yolo']
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return '\n'.join(lines)
