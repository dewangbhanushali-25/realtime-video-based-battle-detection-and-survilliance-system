# realtime-video-based-battle-detection-and-survilliance-system
🐄 Cattle Monitoring System with VLM &amp; Streamlit This project is a real-time cattle activity monitoring system built using OpenCV, Google Gemini Vision Pro (a vision-language model), and Streamlit. It detects cattle in video footage, captures frames at intervals, and sends them to Gemini for semantic scene analysis.





🔍 Features
Real-time cattle detection using OpenCV and a pre-trained model (YOLO or similar)
Vision-Language AI (Gemini) provides detailed understanding of cattle behavior
Smart frame capture: captures one frame every 2 minutes to reduce API usage
Text logging: AI-generated descriptions are saved to gemini_log.txt
Streamlit dashboard for simple viewing and interaction
Frame saving: captured frames are stored in uploads/ for reference




⚙️ How It Works
The system reads video frames in real time.
When cows are detected, it captures a key frame every 2 minutes.
The frame is sent to Gemini Vision Pro for semantic description.
Gemini returns a human-readable scene summary (e.g., "two cows grazing under a tree").
This description is saved in a log file and shown on a Streamlit UI.
🛠️ Tech Stack
Python
OpenCV
Streamlit
Google Gemini Vision API
 cv2 for image processing

 
