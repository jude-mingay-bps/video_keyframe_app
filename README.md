Video to Roboflow: Frame Extraction & Upload Tool
self-hosted web utility to  extract high-quality frames from any video and upload them directly to your Roboflow projects.

✨ Core Features
Multi-Source Video Input: Add videos by pasting a YouTube URL or by uploading a local file (.mp4, .mov, .avi, etc.).

Visual Timeline Scrubber: Don't guess where to clip. A thumbnail-generated timeline provides a high-level overview of the entire video, allowing for precise segment selection.

Precise Segment Selection: Use a draggable and resizable window on the timeline to select the exact video segment you want to process.

Frame-by-Frame Analysis: Once a segment is loaded, step through the extracted frames one by one for careful inspection.

Intuitive Frame Selection: Simply press the spacebar or click the "Toggle Selection" button to choose the perfect frames for your dataset.

Direct Roboflow Integration:

Securely connect to your Roboflow account using your project URL and private API key.

Automatically upload selected frames to your desired project.

Assign a custom batch name for each upload job to keep your datasets organized.

Specify the target dataset split (train, valid, or test) for each batch.

Efficient Workflow: A video processing queue lets you line up multiple videos and process them in a single session.

Keyboard Shortcuts: Navigate, select, and finish processing with keyboard shortcuts (←, →, Space, Enter) for maximum efficiency.

Self-Hosted & Secure: Runs locally on your machine. Your videos and API keys are never exposed to external servers.

Responsive UI: A clean, modern, and responsive interface that works on any screen size.

🚀 Getting Started
Follow these instructions to get the application running on your local machine.

Prerequisites

Python 3.7+

pip (Python package installer)

git (for cloning the repository)

Installation

Clone the Repository
Open your terminal and clone this repository:

Bash
git clone <your-repository-url>
cd <repository-directory>
Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies
A requirements.txt file is needed to install all the necessary Python libraries. Create a file named requirements.txt with the following content:

requirements.txt:

Flask==2.2.2
Flask-Cors==3.0.10
yt-dlp==2023.7.6
opencv-python==4.8.0.74
Werkzeug==2.2.2
requests==2.31.0
numpy==1.24.3
Pillow==9.5.0
Now, install these packages using pip:

Bash
pip install -r requirements.txt
Run the Application
Launch the Flask server with a single command:

Bash
python app.py
You will see output indicating the server is running, typically on http://127.0.0.1:5000.

Open in Browser
Navigate to http://127.0.0.1:5000 in your web browser to start using the application.

🔧 How to Use the Application
The user interface is designed to be intuitive. Follow this workflow for best results:

Configure Roboflow (Required for Upload)

Enter your Roboflow Project URL (e.g., https://app.roboflow.com/workspace-name/project-name).

Enter your Roboflow Private API Key. This is treated like a password and stored only in your browser's local storage.

(Optional) Set a Custom Batch Name. If left empty, the video's filename will be used.

Select the Upload Split (train, valid, or test).

Click Save Configuration. You can also test the connection to ensure your credentials are correct.

Add Videos to the Queue

From YouTube: Paste the full URL of a YouTube video and click "Add YouTube Video".

From Local File: Click "Choose File", select a video from your computer, and click "Upload File".

Add multiple videos to the queue before you begin processing.

Start Processing

Once your queue is ready, click "Start Processing".

The first video's preview and timeline will load. The timeline will be populated with thumbnails from the video.
‚
Select a Video Segment

On the timeline, drag the purple selection window to the desired starting point.

Drag the handles on the left or right side of the window to shorten or lengthen the segment.

Click "Load Frames" to extract all frames from this specific segment.

Select Your Frames

Use the ← and → arrow keys or the "Previous" and "Next" buttons to navigate through the extracted frames.

Press the Spacebar or click "Toggle Selection" to select or deselect a frame. Selected frames will have a green border.

The progress bar at the bottom shows your position in the current segment.

Finish and Upload

Once you have selected all desired frames for the current video, click "Finish This Video".

If Roboflow is configured, your selected frames will be saved locally in the output folder and uploaded directly to your project.

The application will automatically load the next video in the queue.

🗺️ Future Roadmap
This tool is highly functional but has potential for even more features:

[ ] Asynchronous Video Processing: A background worker queue for handling very large video files without blocking the UI.

[ ] Pre-Annotation with a Model: Option to run a baseline model (e.g., YOLO) on the frames and upload them with initial predictions.

[ ] Dockerization: A Dockerfile for easy, one-command deployment.

[ ] User Sessions: Support for multiple users or saving sessions to return to later.

[ ] Additional Export Options: Save frames with annotation metadata in formats like COCO or YOLO TXT.

🤝 Contributing
Contributions are welcome! If you have ideas for new features or improvements, please open an issue to discuss it first. Pull requests are appreciated.