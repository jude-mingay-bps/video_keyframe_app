# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies required by OpenCV and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code into the container
COPY app/ ./app
COPY run.py .
COPY config.py .
COPY yolo11n-pose.pt .


# 7. Create directories for models, uploads, output, and temp files
RUN mkdir -p models uploads output temp

# 8. Expose the port the app runs on
EXPOSE 5000

# 9. Define the command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "run:app"]
