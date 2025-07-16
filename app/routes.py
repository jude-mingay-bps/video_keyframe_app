import os
import uuid
import cv2
import base64
import numpy as np
from datetime import datetime
from flask import jsonify, request, render_template, session, Response, current_app
from werkzeug.utils import secure_filename
from . import utils
from .services import roboflow_handler, yolo_handler, video_processing, movenet_handler

from flask import Blueprint

# Using current_app context to get the app instance
bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/get_model_info')
def get_model_info():
    model, info = yolo_handler.load_yolo_model()
    if model is None:
        return jsonify({'success': False, 'error': 'No YOLO model found in models folder'})
    return jsonify({'success': True, 'model_info': info})


@bp.route('/get_movenet_model_info')
def get_movenet_model_info():
    model, info = movenet_handler.load_movenet_model()
    if model is None:
        return jsonify({'success': False, 'error': 'MoveNet model not available'})
    return jsonify({'success': True, 'model_info': info})


@bp.route('/get_yolo_pose_model_info')
def get_yolo_pose_model_info():
    info = yolo_handler.get_yolo_pose_model_info()
    if info is None:
        return jsonify({'success': False, 'error': 'YOLO pose model not available'})
    return jsonify({'success': True, 'model_info': info})


@bp.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    frame_data = data.get('frame_data')
    confidence = data.get('confidence', 0.25)

    if not frame_data:
        return jsonify({'success': False, 'error': 'No frame data provided'})

    annotations, message = yolo_handler.predict_on_frame(frame_data, confidence)

    if annotations is None:
        return jsonify({'success': False, 'error': message})

    return jsonify({
        'success': True,
        'annotations': annotations,
        'frame_data': frame_data,
        'message': message
    })


@bp.route('/predict_pose_frame', methods=['POST'])
def predict_pose_frame():
    data = request.json
    frame_data = data.get('frame_data')
    confidence = data.get('confidence', 0.3)

    if not frame_data:
        return jsonify({'success': False, 'error': 'No frame data provided'})

    pose_annotations, message = movenet_handler.predict_pose_on_frame(frame_data, confidence)

    if pose_annotations is None:
        return jsonify({'success': False, 'error': message})

    return jsonify({
        'success': True,
        'pose_annotations': pose_annotations,
        'frame_data': frame_data,
        'message': message
    })


@bp.route('/predict_yolo_pose_frame', methods=['POST'])
def predict_yolo_pose_frame():
    data = request.json
    frame_data = data.get('frame_data')
    confidence = data.get('confidence', 0.3)

    if not frame_data:
        return jsonify({'success': False, 'error': 'No frame data provided'})

    pose_annotations, message = yolo_handler.predict_pose_on_frame(frame_data, confidence)

    if pose_annotations is None:
        return jsonify({'success': False, 'error': message})

    return jsonify({
        'success': True,
        'pose_annotations': pose_annotations,
        'frame_data': frame_data,
        'message': message
    })


@bp.route('/get_cached_annotations', methods=['POST'])
def get_cached_annotations():
    data = request.json
    frame_num = data.get('frame_num')
    
    if frame_num is None:
        return jsonify({'success': False, 'error': 'No frame number provided'})
    
    cached_result = video_processing.get_frame_annotations(frame_num)
    
    if cached_result is not None:
        return jsonify({
            'success': True,
            'cached': True,
            'annotations': cached_result.get('annotations'),
            'message': cached_result.get('status', 'From cache'),
            'processed': cached_result.get('processed', True)
        })
    else:
        return jsonify({
            'success': False,
            'cached': False,
            'message': 'No cached annotations found'
        })


@bp.route('/get_cached_pose_annotations', methods=['POST'])
def get_cached_pose_annotations():
    data = request.json
    frame_num = data.get('frame_num')
    
    if frame_num is None:
        return jsonify({'success': False, 'error': 'No frame number provided'})
    
    cached_result = video_processing.get_pose_annotations(frame_num)
    
    if cached_result is not None:
        return jsonify({
            'success': True,
            'cached': True,
            'pose_annotations': cached_result.get('pose_annotations'),
            'message': cached_result.get('status', 'From cache'),
            'processed': cached_result.get('processed', True)
        })
    else:
        return jsonify({
            'success': False,
            'cached': False,
            'message': 'No cached pose annotations found'
        })


@bp.route('/get_background_processing_status', methods=['GET'])
def get_background_processing_status():
    total_frames = len(video_processing.frame_annotations_cache)
    processed_frames = sum(1 for v in video_processing.frame_annotations_cache.values() if v.get('processed'))
    
    return jsonify({
        'success': True,
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'is_complete': total_frames > 0 and processed_frames == total_frames,
        'progress_percentage': (processed_frames / total_frames * 100) if total_frames > 0 else 0
    })


@bp.route('/test_roboflow', methods=['POST'])
def test_roboflow_endpoint():
    data = request.json
    api_key = data.get('api_key')
    project_url = data.get('project_url')

    if not api_key or not project_url:
        return jsonify({'success': False, 'error': 'Missing API key or project URL'})

    success, message = roboflow_handler.test_roboflow_connection(api_key, project_url)
    return jsonify({'success': success, 'message': message})


@bp.route('/get_video_info', methods=['POST'])
def get_video_info():
    data = request.json
    video_id = data.get('video_id')

    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found in session'})

    video_info = session['videos'][video_id]
    video_path = video_info['path']

    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': f'Video file not found'})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'success': False, 'error': 'Cannot open video file'})

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    if duration <= 0:
        return jsonify({'success': False, 'error': 'Invalid video duration'})

    return jsonify({
        'success': True,
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count
    })


@bp.route('/video/<video_id>')
def serve_video(video_id):
    if 'videos' not in session or video_id not in session['videos']:
        return 'Video not found', 404

    video_path = session['videos'][video_id]['path']

    if not os.path.exists(video_path):
        return 'Video file not found', 404

    def generate():
        with open(video_path, 'rb') as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)

    response = Response(generate(), mimetype='video/mp4')
    response.headers['Accept-Ranges'] = 'bytes'
    return response


@bp.route('/add_youtube', methods=['POST'])
def add_youtube():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})

    video_id = str(uuid.uuid4())
    base_path = os.path.join(current_app.config['TEMP_FOLDER'], video_id)

    success, title_or_error = video_processing.download_youtube_video(url, base_path)

    if success:
        video_path = None
        for file in os.listdir(current_app.config['TEMP_FOLDER']):
            if file.startswith(video_id):
                video_path = os.path.join(current_app.config['TEMP_FOLDER'], file)
                break

        if not video_path:
            return jsonify({'success': False, 'error': 'Downloaded file not found'})

        if 'videos' not in session:
            session['videos'] = {}

        session['videos'][video_id] = {
            'path': video_path,
            'name': title_or_error if title_or_error else url,
            'type': 'youtube'
        }
        session.modified = True

        return jsonify({
            'success': True,
            'video': {'id': video_id, 'name': title_or_error if title_or_error else url, 'type': 'youtube'}
        })
    else:
        return jsonify({'success': False, 'error': f'Failed to download: {title_or_error}'})


@bp.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    if file and utils.allowed_file(file.filename):
        video_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{video_id}_{filename}')
        file.save(video_path)

        if 'videos' not in session:
            session['videos'] = {}

        session['videos'][video_id] = {
            'path': video_path,
            'name': filename,
            'type': 'upload'
        }
        session.modified = True

        return jsonify({
            'success': True,
            'video': {'id': video_id, 'name': filename, 'type': 'upload'}
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'})


@bp.route('/extract_frames', methods=['POST'])
def extract_frames_endpoint():
    data = request.json
    video_id = data.get('video_id')
    start_time = data.get('start_time', 0)
    duration = data.get('duration', 30)
    target_fps = data.get('target_fps', 10)
    confidence_threshold = data.get('confidence_threshold', 0.25)

    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})

    video_path = session['videos'][video_id]['path']

    frames = video_processing.extract_frames(video_path, start_time, duration, target_fps, 
                                              start_background_processing=True, 
                                              confidence_threshold=confidence_threshold)

    if frames:
        return jsonify({'success': True, 'frames': frames, 'fps': target_fps, 'frame_count': len(frames)})
    else:
        return jsonify({'success': False, 'error': 'Failed to extract frames'})


@bp.route('/get_timeline_thumbnails', methods=['POST'])
def get_timeline_thumbnails_endpoint():
    data = request.json
    video_id = data.get('video_id')

    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})

    video_path = session['videos'][video_id]['path']
    thumbnails = video_processing.extract_timeline_thumbnails(video_path)

    if thumbnails is not None:
        return jsonify({'success': True, 'thumbnails': thumbnails})
    else:
        return jsonify({'success': False, 'error': 'Failed to extract timeline thumbnails'})


@bp.route('/save_frames', methods=['POST'])
def save_frames():
    data = request.json
    video_id = data.get('video_id')
    frames_data = data.get('frames', [])
    upload_to_roboflow = data.get('upload_to_roboflow', False)
    roboflow_config = data.get('roboflow_config', {})

    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})

    _, model_info_loaded = yolo_handler.load_yolo_model()
    label_map_for_upload = model_info_loaded['classes'].copy() if model_info_loaded else {}

    video_info = session['videos'][video_id]
    video_name_raw = os.path.splitext(video_info['name'])[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_app.config['OUTPUT_FOLDER'], f'{video_name_raw}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    roboflow_results = []

    has_corrections = any(
        'predictions' in fd and fd['predictions'] and any(
            ann.get('was_corrected') for ann in fd['predictions'].get('annotations', [])
        ) for fd in frames_data
    )
    if has_corrections:
        label_map_for_upload[999] = 'Other'

    for i, frame_data in enumerate(frames_data):
        frame_index = frame_data.get('frameIndex', i)

        frame_bytes = base64.b64decode(frame_data['data'])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        image_base_name = f'video_{video_name_raw}_frame_{i + 1:03d}_time_{frame_data["time"]:.1f}s'
        image_filename_png = f'{image_base_name}.png'
        filepath = os.path.join(output_dir, image_filename_png)
        cv2.imwrite(filepath, frame)

        annotation_data = None
        if 'predictions' in frame_data and frame_data['predictions']:
            predictions = frame_data['predictions']
            if predictions and 'annotations' in predictions and predictions['annotations']:
                annotations = predictions['annotations']
                img_height, img_width = frame.shape[:2]
                annotation_data = yolo_handler.create_yolo_annotation_file(annotations, img_width, img_height)

                annotation_filename = f'{image_base_name}.txt'
                annotation_filepath = os.path.join(output_dir, annotation_filename)
                with open(annotation_filepath, 'w') as f:
                    f.write(annotation_data)

        if upload_to_roboflow and roboflow_config.get('apiKey') and roboflow_config.get('url'):
            image_name_for_upload = f'{image_base_name}.jpg'

            base_batch_name = roboflow_config.get('batchName') or video_name_raw
            batch_name_for_upload = f"{base_batch_name} (Auto labeled Please review)"

            split = roboflow_config.get('split', 'train')

            success, message = roboflow_handler.upload_to_roboflow_api(
                roboflow_config['apiKey'], roboflow_config['url'], frame_data['data'],
                image_name_for_upload, split=split, batch_name=batch_name_for_upload,
                annotation_data=annotation_data, label_map=label_map_for_upload
            )

            roboflow_results.append({
                'frame': i, 'frame_index': frame_index, 'success': success,
                'message': message, 'with_annotations': annotation_data is not None
            })

    response_data = {'success': True, 'output_dir': output_dir, 'frame_count': len(frames_data)}

    if roboflow_results:
        response_data['roboflow_results'] = roboflow_results

    return jsonify(response_data)


@bp.route('/cleanup', methods=['POST'])
def cleanup():
    if 'videos' in session:
        for video_id, video_info in session['videos'].items():
            if video_info['type'] == 'youtube' and os.path.exists(video_info['path']):
                os.remove(video_info['path'])
        session.pop('videos', None)

    return jsonify({'success': True})


# Register the blueprint in the app factory
def create_app(config_class=None):
    # ... (app creation logic from __init__.py)
    from flask import Flask
    from config import Config

    app = Flask(__name__)
    app.config.from_object(Config)

    from .main import bp as main_bp
    app.register_blueprint(main_bp)

    return app
