# realtime-video-based-battle-detection-and-survilliance-system
🐄 Cattle Monitoring System with VLM &amp; Streamlit This project is a real-time cattle activity monitoring system built using OpenCV, Google Gemini Vision Pro (a vision-language model), and Streamlit. It detects cattle in video footage, captures frames at intervals, and sends them to Gemini for semantic scene analysis.



🔍 Features

Real-time Detection: Identifies various cattle types (cows, sheep, buffalo, goats, pigs) in video streams
Activity Recognition: AI-powered analysis of what the animals are doing in each frame
Dual Input Support: Process pre-recorded videos or connect directly to webcam feeds
Smart Processing: Configurable frame skipping for performance optimization
Interactive Dashboard: Built with Streamlit for intuitive monitoring and control
Activity Logging: Comprehensive record of detected animals and their behaviors
High Accuracy: Leverages YOLOv8e for reliable detection in various conditions


Computer Vision: OpenCV, YOLO11 object detection
AI Integration: Google Gemini Vision Pro for semantic scene understanding
Frontend: Streamlit interactive dashboard
Environment Management: Python dotenv for configuration


⚙️ How It Works

Detection Pipeline: Video frames are processed through YOLO11 to detect and classify cattle
Crop & Analyze: Detected animals are cropped from the frame and sent to Gemini Vision
AI Description: Gemini analyzes each animal's posture, position, and activity
Display & Log: Results are displayed in real-time and stored for analysis 

git clone https://github.com/yourusername/cattle-detection.git
cd cattle-detection
# Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
download yolo11.pt model using code in ipynb 
streamlit run app.py


cattle-detection/
├── app.py                # Main Streamlit application
├── cattledetection.py    # YOLO-based detection module
├── vlm.py                # Gemini Vision integration
├── .env                  # API keys and configuration
└── uploads/              # Folder for analyzed frames





 
