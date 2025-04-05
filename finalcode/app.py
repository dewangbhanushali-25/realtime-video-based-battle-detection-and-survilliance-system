import streamlit as st
import cv2
import tempfile
import os
import uuid
import time
from pathlib import Path
from detection import detect_cattle
from vlm import describe_with_gemini
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Make sure to check if it loaded correctly
if not GEMINI_API_KEY:
    raise ValueError(" Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")


# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
last_gemini_call = 0
GEMINI_INTERVAL = 120  # 2 minutes in seconds

# Streamlit UI
st.set_page_config(page_title="Cattle Detection & Activity Recognition", layout="wide")
st.title("Real-time Cattle Detection & Activity Recognition")

# Sidebar for settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
skip_frames = st.sidebar.slider("Process every N frames", 1, 30, 10)
enable_vlm = st.sidebar.checkbox("Enable Activity Recognition with VLM", value=True)

# Input source selection
input_source = st.radio("Select Input Source:", ["Video File", "Webcam"])

#@st.cache_data(show_spinner=False)
import time

# Globals
last_gemini_call = 0
last_gemini_response = None
GEMINI_INTERVAL = 120  # 2 minutes

def handle_detection(image_path):
    global last_gemini_call, last_gemini_response

    current_time = time.time()

    if current_time - last_gemini_call >= GEMINI_INTERVAL:
        print(" Calling Gemini for description...")

        try:
            description = describe_with_gemini(image_path)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Update last call time and response
            last_gemini_call = current_time
            last_gemini_response = description

            # Log result to file
            with open("gemini_log.txt", "a") as log_file:
                log_file.write(f"[{timestamp}]  Gemini: {description}\n")

            print(f" Gemini result at {timestamp}: {description}")
            return description

        except Exception as e:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            error_msg = f"[{timestamp}]  Error calling Gemini: {e}"

            # Log error to file
            with open("gemini_log.txt", "a") as log_file:
                log_file.write(error_msg + "\n")

            print(error_msg)
            return f"Error analyzing: {str(e)[:50]}"

    else:
        remaining = int(GEMINI_INTERVAL - (current_time - last_gemini_call))
        print(f"⏳ Skipping Gemini call — wait {remaining} seconds.")
        return last_gemini_response


# Function to process video (shared between file and webcam)
def process_video(video_source, is_webcam=False):
    global last_gemini_call

    # Create a progress bar (for file only)
    if not is_webcam:
        progress_bar = st.progress(0)
        status_text = st.empty()

    # Create display areas
    col1, col2 = st.columns([3, 1])
    video_display = col1.empty()
    info_display = col2.empty()

    # Detection results storage
    all_cattle_data = []
    frame_count = 0
    start_time = time.time()
    last_processed_frame_time = 0
    current_frame_description = None

    # Get total frames for progress (file only)
    total_frames = 1
    if not is_webcam:
        vcap = cv2.VideoCapture(video_source)
        if vcap.isOpened():
            total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vcap.get(cv2.CAP_PROP_FPS)
            st.info(f"Video loaded: {fps:.1f} FPS, {total_frames} frames")
        vcap.release()

    # Create stop button for webcam
    stop_button_placeholder = st.empty()
    stop_button = stop_button_placeholder.button("Stop Processing") if is_webcam else None

    try:
        # Process video frames
        for frame, cattle_data in detect_cattle(video_source, confidence_threshold, skip_frames, is_webcam=is_webcam):
            if is_webcam and stop_button_placeholder.button("Stop Processing", key="stop_webcam"):
                break

            frame_count += 1

            # Update progress (file only)
            if not is_webcam:
                progress = min(frame_count * skip_frames / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count*skip_frames}/{total_frames}")

            current_frame_description = None
            if enable_vlm and cattle_data:
                current_time = time.time()
                if current_time - last_gemini_call >= GEMINI_INTERVAL:
                    print("Attempting Gemini call for current frame...")
                    # Save the first detected cattle's crop for description
                    crop_filename = f"frame_{frame_count}_cattle_0_{uuid.uuid4().hex[:6]}.jpg"
                    crop_path = str(UPLOAD_DIR / crop_filename)
                    cv2.imwrite(crop_path, cattle_data[0]["crop"]) # Use the first detected cattle
                    current_frame_description = handle_detection(crop_path)

                current_data = []
                for i, cattle in enumerate(cattle_data):
                    cattle["frame_id"] = frame_count * skip_frames
                    if current_frame_description:
                        cattle["action"] = current_frame_description
                    else:
                        cattle["action"] = "Analyzing..." if enable_vlm else "Not enabled"
                    current_data.append(cattle)

                # Update all data
                all_cattle_data.extend(current_data)

                # Display info about detected cattle in the current frame
                info_display.markdown("### Detected Cattle")
                for cattle in current_data:
                    info_display.markdown(f"**{cattle['type']}**: {cattle['action']}")

            # Display frame
            video_display.image(frame, caption="Detection Results", channels="BGR", use_column_width=True)

        # Show summary (for file only, webcam doesn't need this)
        if not is_webcam:
            processing_time = time.time() - start_time
            st.success(f"Processing complete! Detected {len(all_cattle_data)} cattle instances in {processing_time:.2f} seconds")

            # Show summary of activities (optional)
            if all_cattle_data:
                st.subheader("Activity Summary")
                for cattle_type in set(c["type"] for c in all_cattle_data):
                    st.write(f"**{cattle_type}** activities:")
                    activities = [c["action"] for c in all_cattle_data if c["type"] == cattle_type]
                    for activity in set(activities)[:5]:  # Show top 5 unique activities
                        st.write(f"- {activity}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    finally:
        # Clean up temporary file if using video file
        if not is_webcam and 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)

# Handle video file input
if input_source == "Video File":
    st.subheader("Upload Video")
    video_file = st.file_uploader("Upload a cattle video", type=["mp4", "avi", "mov"])

    if video_file:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        try:
            # Video info
            vcap = cv2.VideoCapture(video_path)
            if not vcap.isOpened():
                st.error("Error: Could not open video file. Please try another file.")
            else:
                vcap.release()

                if st.button("Start Detection"):
                    last_gemini_call = 0 # Reset timer when starting new processing
                    process_video(video_path, is_webcam=False)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Handle webcam input
else:  # Webcam
    st.subheader("Webcam Settings")
    camera_id = st.selectbox("Select Camera", options=list(range(3)), index=0)

    if st.button("Start Webcam Detection"):
        last_gemini_call = 0 # Reset timer when starting new processing
        process_video(camera_id, is_webcam=True)

# Footer
st.markdown("---")
st.markdown("Cattle Detection and Activity Recognition System")
# Footer
# Divide the layout into two columns
# Footer
# Divide the layout into two columns
left_col, right_col = st.columns([1, 2])  # Adjust ratios as needed

# Left column: Gemini Log
with left_col:
    st.markdown("Gemini Log")
    log_file_path = "gemini_log.txt"

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as f:
            log_contents = f.read()
        st.text_area("Log Output", log_contents, height=500)
    else:
        st.info("No log file found yet.")

# Right column: Reserved for future features
with right_col:
    st.markdown("### Additional Features Coming Soon")
    st.write("This space will display future analysis or insights from VLM.")

#  Section below both columns
st.markdown("---")
st.subheader("System Notes & Debug Info")

