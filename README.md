# Video to Roboflow: Frame Extraction & Upload Tool

A self-hosted web utility to effortlessly extract high-quality frames from any video and upload them directly to your Roboflow projects. This tool streamlines the data annotation pipeline by bridging the gap between raw video footage and a ready-to-label dataset.

* **The Problem**: Manually scrubbing through videos, capturing screenshots, and uploading them one-by-one is tedious and inefficient.
* **The Solution**: This tool provides a unified interface to select precise video segments, choose the best frames, and upload them directly to Roboflow in organized batches.

---

## ✨ Core Features

| Feature                    | Description                                                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📺 Multi-Source Input** | Add videos by pasting a **YouTube URL** or uploading local files (`.mp4`, `.mov`, `.avi`, etc.).                                                            |
| **🎞️ Visual Timeline Scrubber** | A thumbnail-generated timeline provides a bird's-eye view of the video, allowing for precise clip selection without guesswork.                             |
| **✂️ Precise Segmenting** | Use a draggable and resizable window on the timeline to isolate the exact video segment you want to process.                                            |
| **🖼️ Frame-by-Frame Analysis** | Step through extracted frames one-by-one for careful inspection and selection.                                                                        |
| **🖱️ Intuitive Selection** | Simply press the **spacebar** or click to select/deselect the perfect frames for your dataset. Selected frames are highlighted with a green border. |
| **🤖 Direct Roboflow Upload** | Securely connect to your Roboflow account to upload selected frames directly to your project and desired dataset split (train, valid, or test).      |
| **🗂️ Organized Batches** | Assign a custom batch name for each upload job to keep your Roboflow datasets neatly organized.                                                          |
| **⚡ Efficient Workflow** | A video processing queue lets you line up multiple videos and process them in a single, uninterrupted session.                                        |
| **⌨️ Keyboard Shortcuts** | Navigate (`←`, `→`), select (`Space`), and finish (`Enter`) with keyboard shortcuts for maximum efficiency.                                               |
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
    git clone <your-repository-url>
    cd <repository-directory>
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
    *(Note: Ensure you have a `requirements.txt` file in your repository with the content you provided.)*

4.  **Run the Application**
    Launch the Flask server:
    ```bash
    python app.py
    ```
    The server will start, typically on `http://127.0.0.1:5000`.

5.  **Open in Browser**
    Navigate to **<http://127.0.0.1:5000>** in your web browser to start using the tool.

---

## 🔧 How to Use the Application

The user interface is designed to be intuitive. For best results, follow this workflow:

1.  **Configure Roboflow (Required for Upload)**
    * Enter your **Roboflow Project URL** and **Private API Key**. Your key is treated like a password and stored only in your browser's local storage.
    * (Optional) Set a **Custom Batch Name**. If left empty, the video's filename is used.
    * Select the **Upload Split** (`train`, `valid`, or `test`).
    * Click **Save Configuration**. You can test the connection to ensure your credentials are correct.
    `[Image: Screenshot of the Roboflow configuration section]`

2.  **Add Videos to the Queue**
    * **From YouTube**: Paste a video URL and click **Add YouTube Video**.
    * **From Local File**: Click **Choose File**, select a video, and click **Upload File**.
    * Add as many videos as you need before processing.

3.  **Start Processing & Select a Segment**
    * Click **Start Processing** to load the first video.
    * On the timeline, drag the purple selection window and resize it to define your clip.
    * Click **Load Frames** to extract all frames from this segment.
    `[Image: Screenshot of the timeline scrubber and segment selection]`

4.  **Select Your Frames**
    * Navigate through the frames using the `←` and `→` arrow keys or the on-screen buttons.
    * Press the `Spacebar` or click **Toggle Selection** to select or deselect a frame.
    `[Image: Screenshot of the frame-by-frame view with a selected frame]`

5.  **Finish and Upload**
    * Once you're done selecting frames for a video, click **Finish This Video**.
    * Your selected frames will be uploaded directly to your configured Roboflow project.
    * The application will automatically load the next video in your queue.

---

## 🗺️ Future Roadmap

This tool is highly functional, but here are some potential features and improvements for the future:

* [ ] **Asynchronous Video Processing**: Implement a background worker queue for handling very large video files without blocking the UI.
* [ ] **Pre-Annotation with a Model**: Add an option to run a baseline model (e.g., YOLO) on frames and upload them with initial predictions.
* [ ] **Dockerization**: Provide a `Dockerfile` for easy, one-command deployment.
* [ ] **User Sessions**: Support multiple users or saving sessions to return to later.
* [ ] **Additional Export Options**: Allow saving frames with annotation metadata in formats like COCO or YOLO TXT.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or improvements, please open an issue to discuss it first. Pull requests that follow the project's coding style are highly appreciated.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). *(Replace MIT License with your actual license)*
