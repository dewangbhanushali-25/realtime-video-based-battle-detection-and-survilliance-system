#🐄 Real-Time Video-Based Cattle Activity Detection & Surveillance System

A real-time cattle behavior detection and surveillance system built using OpenCV, YOLOv11, Google Gemini Flash 2.0, and Streamlit. This intelligent system monitors cattle behavior in real-time and provides AI-powered semantic understanding of their activity.


## 🔍 Features

- ✅ **Real-time Cattle Detection**  
  Detects and classifies cows, buffaloes, goats, sheep, and pigs using **YOLOv11**.

- 🤖 **AI-Powered Activity Recognition**  
  Uses **Gemini Flash 2.0** for high-speed vision-language scene understanding to analyze posture and activity.

- 🎥 **Flexible Video Input**  
  Supports live webcam feed and pre-recorded video uploads.

- ⚡ **Configurable Frame Skipping**  
  Boost performance by skipping frames dynamically.

- 📊 **Interactive Streamlit Dashboard**  
  Control, view, and analyze results in a clean, real-time interface.

- 📁 **Activity Logging & Frame Storage**  
  Automatically logs animal activity and saves analyzed frames.

---

## 🛠️ Tech Stack

| Layer              | Tools Used                              |
|--------------------|------------------------------------------|
| **Computer Vision**| OpenCV, YOLOv11                          |
| **AI/VLM**         | Google Gemini Flash 2.0                  |
| **Frontend**       | Streamlit                                |
| **Environment**    | Python `dotenv` for config and secrets   |

---

## ⚙️ How It Works

1. **Frame Capture**  
   Input from webcam or uploaded video is read using OpenCV.

2. **YOLOv11 Detection**  
   Objects in each frame are detected and classified (cattle types).

3. **Cropping + VLM Analysis**  
   Detected animals are cropped and sent to **Gemini Flash 2.0**, which semantically understands:
   - Posture (e.g. standing, walking, lying down)
   - Behavior (e.g. grazing, fighting, running)

4. **Display & Logging**  
   - Results displayed in real-time via Streamlit  
   - Each detection is logged and saved

---

## 📁 Project Structure

cattle-detection/ ├── app.py # Main Streamlit app ├── cattledetection.py # YOLOv11 detection logic ├── vlm.py # Gemini Flash 2.0 integration ├── .env # API key and config └── uploads/ # Saved analyzed frames

# how to run the program
- in folder use .env to manage api (here GEMINI_API_KEY is variable of api)
- then use streamlit run app.py in specfic venv of python where streamlit is installed
- you will taken to a browser, input your video by selecting video and start detect
- this will generate detection and log of detection and vlm answer , the api is  called after 3 mins making it cost and time efficent


 
