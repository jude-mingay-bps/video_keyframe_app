# Video to Roboflow: Frame Extraction & Upload Tool

A self-hosted web utility to effortlessly extract high-quality frames from any video and upload them directly to your Roboflow projects. This tool streamlines the data annotation pipeline by bridging the gap between raw video footage and a ready-to-label dataset.

* **The Problem**: Manually scrubbing through videos, capturing screenshots, and uploading them one-by-one is tedious and inefficient.
* **The Solution**: This tool provides a unified interface to select precise video segments, choose the best frames, and upload them directly to Roboflow in organized batches.

---

## ✨ Core Features

| Feature                       | Description                                                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📺 Multi-Source Input** | Add videos by pasting a **YouTube URL** or uploading local files (`.mp4`, `.mov`, `.avi`, etc.).                                                            |
| **🤖 Integrated YOLO Predictions** | Run a local YOLO model on frames within the browser. Bounding boxes are drawn instantly, and annotations are uploaded directly to Roboflow. |
| **🎞️ Visual Timeline Scrubber** | A thumbnail-generated timeline provides a bird's-eye view of the video, allowing for precise clip selection without guesswork.                             |
| **✂️ Precise Segmenting** | Use a draggable and resizable window on the timeline to isolate the exact video segment you want to process.                                            |
| **🖼️ Frame-by-Frame Analysis** | Step through extracted frames one-by-one for careful inspection and selection.                                                                        |
| **🖱️ Intuitive Selection** | Simply press the **spacebar** or click to select/deselect the perfect frames for your dataset. Selected frames are highlighted with a green border. |
| **🤖 Direct Roboflow Upload** | Securely connect to your Roboflow account to upload selected frames directly to your project and desired dataset split (train, valid, or test).      |
| **🗂️ Organized Batches** | Assign a custom batch name for each upload job to keep your Roboflow datasets neatly organized.                                                          |
| **⚡ Efficient Workflow** | A video processing queue lets you line up multiple videos and process them in a single, uninterrupted session.                                        |
| **⌨️ Keyboard Shortcuts** | Navigate (`←`, `→`), select (`Space`), predict (`P`), and finish (`Enter`) with keyboard shortcuts for maximum efficiency.                                               |
| **🔒 Self-Hosted & Secure** | Runs locally on your machine. Your videos and private Roboflow API keys are never exposed to external servers.                                         |
| **📱 Responsive UI** | A clean, modern interface that works beautifully on any screen size.                                                                                    |

---

## 🚀 Getting Started

Follow these instructions to get the application running on your local machine.

### Prerequisites

* Python 3.7+
* `pip` (Python package installer)
* `git` (for cloning the repository)

### Installation

1.  **Clone the Repository**
    Open your terminal and run the following commands:
    ```bash
    git clone git@github.com:jude-mingay-bps/video_keyframe_app.git
    cd video_keyframe_app
    ```

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

    * **Windows:**
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all the necessary Python libraries. Install them with a single command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    Launch the Flask server:
    ```bash
    python app.py
    ```
    The server will start, typically on `http://127.0.0.1:5000`.

5.  **Open in Browser**
    Navigate to **http://127.0.0.1:5000** in your web browser to start using the tool.

---
## 🐳 Docker Deployment (Alternative)

For a quick and isolated setup, you can run the application inside a Docker container. This is the recommended method for avoiding dependency conflicts.

### Prerequisites
* [Docker](https://www.docker.com/get-started) installed and running on your machine.

### Instructions
1.  **Place Your Model (Optional)**
    If you want to use the YOLO prediction feature, create a `models` directory in the project folder and place your model file (e.g., `yolov8s.pt`) inside it.

2.  **Build the Docker Image**
    Open your terminal in the project's root directory and run:
    ```bash
    docker build -t video-to-roboflow .
    ```

3.  **Run the Docker Container**
    This command starts the application and connects your local `models`, `output`, and `uploads` folders to the container for persistent storage.
    ```bash
    docker run -d -p 5000:5000 \
      -v "$(pwd)/models:/app/models" \
      -v "$(pwd)/output:/app/output" \
      -v "$(pwd)/uploads:/app/uploads" \
      --name roboflow-uploader \
      video-to-roboflow
    ```

4.  **Open in Browser**
    Navigate to **http://localhost:5000** in your web browser.

---

## 🔧 How to Use the Application

The user interface is designed to be intuitive. For best results, follow this workflow:

1.  **Configure Roboflow & YOLO**
    * **Roboflow**: Enter your **Project URL** and **API Key**. Click **Save Configuration**.
    * **YOLO**: The application will automatically detect any models in your `models` folder. To use them, **enable the prediction toggle switch** and adjust the confidence threshold as needed.

2.  **Add Videos to the Queue**
    * **From YouTube**: Paste a video URL and click **Add YouTube Video**.
    * **From Local File**: Click **Choose File**, select a video, and click **Upload File**.

3.  **Start Processing & Select a Segment**
    * Click **Start Processing** to load the first video.
    * On the timeline, drag the purple selection window and resize it to define your clip.
    * Click **Load Frames** to extract all frames from this segment.

4.  **Select and Predict on Frames**
    * Navigate through frames using the `←` and `→` arrow keys.
    * Press the `Spacebar` to select/deselect a frame.
    * With the prediction toggle enabled, press the `P` key or click **Run Prediction** to generate annotations for the current frame. The frame will be updated with bounding boxes.

5.  **Finish and Upload**
    * Once you're done, click **Finish This Video**.
    * Selected frames are uploaded to Roboflow. If a frame has YOLO predictions, its corresponding `.txt` annotation file will be **uploaded automatically** with it.
    * The application loads the next video in your queue.

---

## 🗺️ Future Roadmap

This tool is highly functional, but here are some potential features and improvements for the future:

* [ ] **Asynchronous Video Processing**: Implement a background worker queue for handling very large video files without blocking the UI.
* [ ] **User Sessions**: Support multiple users or saving sessions to return to later.
* [ ] **Additional Export Options**: Allow saving frames with annotation metadata in formats like COCO.
* [x] **Dockerization**: Provide a `Dockerfile` for easy, one-command deployment.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or improvements, please open an issue to discuss it first. Pull requests that follow the project's coding style are highly appreciated.

---

## 📜 License

This project is licensed under the MIT License.