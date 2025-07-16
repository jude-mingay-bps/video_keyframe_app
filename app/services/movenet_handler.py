import os
import cv2
import base64
import numpy as np
from config import Config

# Global variables for MoveNet model
loaded_movenet_model = None
movenet_model_info = None

# MoveNet keypoint names (17 keypoints in COCO format)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# MoveNet skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # legs
]

def load_movenet_model():
    """Load MoveNet Lightning model for pose estimation."""
    global loaded_movenet_model, movenet_model_info
    
    if loaded_movenet_model is not None:
        return loaded_movenet_model, movenet_model_info
    
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        print("Loading MoveNet Lightning model...")
        
        # Load MoveNet Lightning model from TensorFlow Hub
        model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        loaded_movenet_model = hub.load(model_url)
        
        movenet_model_info = {
            'name': 'MoveNet Lightning',
            'type': 'pose_estimation',
            'description': 'Ultra fast and accurate pose detection model',
            'keypoints': 17,
            'input_size': 192,
            'keypoint_names': KEYPOINT_NAMES,
            'skeleton_connections': SKELETON_CONNECTIONS
        }
        
        print(f"MoveNet model loaded successfully: {movenet_model_info['name']}")
        return loaded_movenet_model, movenet_model_info
        
    except ImportError as e:
        print(f"MoveNet dependencies not available: {e}")
        # Return fallback info
        movenet_model_info = {
            'name': 'MoveNet Lightning (Not Available)',
            'type': 'pose_estimation',
            'description': 'Install tensorflow and tensorflow-hub to enable MoveNet pose estimation',
            'keypoints': 17,
            'input_size': 192,
            'keypoint_names': KEYPOINT_NAMES,
            'skeleton_connections': SKELETON_CONNECTIONS
        }
        return None, movenet_model_info
        
    except Exception as e:
        print(f"Error loading MoveNet model: {e}")
        return None, None


def preprocess_image(image_data):
    """Preprocess image for MoveNet inference."""
    try:
        import tensorflow as tf
        
        # Decode image
        if isinstance(image_data, str):
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # Already decoded image
            image = image_data
        
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to MoveNet input size (192x192)
        image = cv2.resize(image, (192, 192))
        
        # Convert to tensor and normalize
        image = tf.cast(image, dtype=tf.int32)
        image = tf.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_pose_on_frame(frame_base64, confidence_threshold=0.3):
    """Run MoveNet pose prediction on a frame."""
    model, info = load_movenet_model()
    
    if model is None:
        return None, "MoveNet model not available - install tensorflow and tensorflow-hub"
    
    try:
        import tensorflow as tf
        
        # Preprocess image
        input_image = preprocess_image(frame_base64)
        if input_image is None:
            return None, "Failed to preprocess image"
        
        # Run inference
        outputs = model.signatures['serving_default'](input_image)
        keypoints = outputs['output_0'].numpy()
        
        # Process keypoints
        pose_annotations = []
        
        # MoveNet returns keypoints in shape (1, 1, 17, 3) where:
        # - 1st dimension: batch size
        # - 2nd dimension: number of detected poses (always 1 for single pose)
        # - 3rd dimension: number of keypoints (17)
        # - 4th dimension: [y, x, confidence]
        
        keypoints = keypoints[0, 0, :, :]  # Remove batch and pose dimensions
        
        # Convert to our format
        valid_keypoints = []
        
        # Decode the original image to get actual dimensions
        image_bytes = base64.b64decode(frame_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        original_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if original_frame is not None:
            original_height, original_width = original_frame.shape[:2]
        else:
            original_height, original_width = 192, 192
        
        for i, (y, x, conf) in enumerate(keypoints):
            if conf >= confidence_threshold:
                valid_keypoints.append({
                    'name': info['keypoint_names'][i],
                    'index': i,
                    'x': float(x * original_width),   # Scale to original image size
                    'y': float(y * original_height),  # Scale to original image size
                    'confidence': float(conf)
                })
        
        if valid_keypoints:
            pose_annotations.append({
                'person_id': 0,
                'keypoints': valid_keypoints,
                'num_keypoints': len(valid_keypoints),
                'avg_confidence': float(np.mean([kpt['confidence'] for kpt in valid_keypoints]))
            })
        
        return pose_annotations, "Success"
        
    except Exception as e:
        print(f"MoveNet prediction error: {e}")
        return None, f"Prediction error: {str(e)}"


def create_pose_annotation_data(pose_annotations):
    """Create pose annotation data in a structured format."""
    annotation_data = []
    
    for person in pose_annotations:
        person_data = {
            'person_id': person['person_id'],
            'keypoints': person['keypoints'],
            'num_keypoints': person['num_keypoints'],
            'avg_confidence': person['avg_confidence']
        }
        annotation_data.append(person_data)
    
    return annotation_data


def draw_pose_on_frame(frame, pose_annotations, confidence_threshold=0.3):
    """Draw pose keypoints and skeleton on frame."""
    if not pose_annotations:
        return frame
    
    # Colors for visualization
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    annotated_frame = frame.copy()
    
    for person_idx, person in enumerate(pose_annotations):
        color = colors[person_idx % len(colors)]
        
        # Create keypoint lookup
        keypoint_dict = {}
        for kpt in person['keypoints']:
            if kpt['confidence'] >= confidence_threshold:
                keypoint_dict[kpt['index']] = (int(kpt['x']), int(kpt['y']))
        
        # Draw skeleton connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx in keypoint_dict and end_idx in keypoint_dict:
                pt1 = keypoint_dict[start_idx]
                pt2 = keypoint_dict[end_idx]
                cv2.line(annotated_frame, pt1, pt2, color, 2)
        
        # Draw keypoints
        for kpt in person['keypoints']:
            if kpt['confidence'] >= confidence_threshold:
                x, y = int(kpt['x']), int(kpt['y'])
                cv2.circle(annotated_frame, (x, y), 4, color, -1)
                
                # Draw keypoint label
                cv2.putText(annotated_frame, kpt['name'], (x + 6, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw person info
        if person['keypoints']:
            first_kpt = person['keypoints'][0]
            cv2.putText(annotated_frame, f"Person {person_idx + 1}", 
                       (int(first_kpt['x']), int(first_kpt['y']) - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame


def get_model_info():
    """Get MoveNet model information."""
    _, info = load_movenet_model()
    return info