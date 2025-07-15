import cv2
import yt_dlp
import base64
from moviepy.video.io.VideoFileClip import VideoFileClip


def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp."""
    if output_path.endswith('.mp4'):
        output_path = output_path[:-4]

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path + '.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return True, info.get('title', 'YouTube Video')
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return False, str(e)


def extract_frames(video_path, start_time, duration=30, target_fps=10):
    """Extract frames from video at specified fps using moviepy."""
    try:
        with VideoFileClip(video_path) as video:
            end_time = min(start_time + duration, video.duration)

            frames = []
            current_time = start_time
            frame_interval = 1.0 / target_fps

            while current_time < end_time:
                frame_array = video.get_frame(current_time)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                frame_num = int(current_time * video.fps)

                frames.append({
                    'data': frame_base64,
                    'frame_num': frame_num,
                    'time': current_time
                })

                current_time += frame_interval

        return frames

    except Exception as e:
        print(f"Error extracting frames with moviepy: {e}")
        return None


def extract_timeline_thumbnails(video_path, num_thumbnails=20):
    """Extract a set of thumbnails for the entire video timeline."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return []

    thumbnails = []
    frame_interval = max(1, frame_count // num_thumbnails)

    for i in range(num_thumbnails):
        frame_num = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        height = 90
        if frame.shape[0] == 0:
            continue

        aspect_ratio = frame.shape[1] / frame.shape[0]
        width = int(height * aspect_ratio)
        resized_frame = cv2.resize(frame, (width, height))

        _, buffer = cv2.imencode('.jpg', resized_frame)
        thumb_base64 = base64.b64encode(buffer).decode('utf-8')
        thumbnails.append(thumb_base64)

    cap.release()
    return thumbnails
