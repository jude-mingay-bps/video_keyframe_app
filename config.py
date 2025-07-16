import os


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

    # Folder configurations
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'output'
    TEMP_FOLDER = 'temp'
    MODELS_FOLDER = 'models'

    # Allowed file extensions for video uploads
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

    # Maximum file size for uploads (500MB)
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year cache for static files
    THREADS_PER_PAGE = 8  # Number of threads for processing
    
    # Frame extraction settings
    FRAME_EXTRACTION_THREADS = 8  # Parallel frame extraction
    FRAME_JPEG_QUALITY = 70  # Lower quality for faster loading
    
    # Create directories if they don't exist
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, MODELS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

