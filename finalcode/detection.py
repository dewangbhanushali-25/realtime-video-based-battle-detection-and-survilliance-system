import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

# Define cattle classes
CATTLE_CLASSES = ["sheep", "cow", "buffalo", "goat", "pig"]

def load_model(model_path="/home/dew/intership/project1/main/yolo11n.pt"):
    """
    Load the YOLO model with error handling
    """
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            # Use pre-trained model if custom model doesn't exist
            print(f"Model {model_path} not found, using default YOLO model")
            model = YOLO("/home/dew/intership/project1/main/yolo11n.pt.pt")  # Default to YOLOv8 nano
        else:
            model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load YOLO model: {str(e)}")

def detect_cattle(video_source, conf_threshold=0.5, skip_frames=1, is_webcam=False):
    """
    Detect cattle in video with frame skipping for better performance
    
    Args:
        video_source: Path to video file or camera ID
        conf_threshold: Confidence threshold for detection
        skip_frames: Process every N frames for performance
        is_webcam: Whether video_source is a webcam
    """
    # Load model
    model = load_model()
    
    # Open video source (file or webcam)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise Exception(f"Cannot open video source: {video_source}")
    
    # Set webcam properties if using webcam
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every N frames for performance
            frame_count += 1
            if not is_webcam and frame_count % skip_frames != 0:
                continue
            
            # For webcam, we might want to process every frame or use a smaller skip
            if is_webcam and frame_count % max(1, skip_frames // 2) != 0:
                continue
            
            # Run detection
            results = model(frame, conf=conf_threshold)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            cattle_data = []
            
            # Process detected objects
            for i, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
                # Check if the class is a cattle type
                class_idx = int(class_id)
                if class_idx in [0, 16, 19, 20, 21]:  # COCO classes for animal types
                    # Map COCO class to cattle type (simplified mapping)
                    if class_idx == 16:  # COCO dog - treat as general livestock
                        cattle_type = "livestock"
                    elif class_idx == 19:  # COCO cow
                        cattle_type = "cow"
                    elif class_idx == 20:  # COCO sheep
                        cattle_type = "sheep"
                    elif class_idx == 21:  # COCO elephant - can be buffalo
                        cattle_type = "buffalo"
                    else:
                        cattle_type = "other_animal"
                elif 0 <= class_idx < len(CATTLE_CLASSES):  # If using custom model with specific classes
                    cattle_type = CATTLE_CLASSES[class_idx]
                else:
                    continue  # Skip non-cattle objects
                
                # Extract bounding box
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Skip invalid crops
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                # Extract cropped image
                crop = frame[y1:y2, x1:x2].copy()
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cattle_type} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Store detected cattle info
                cattle_data.append({
                    "type": cattle_type,
                    "confidence": float(confidence),
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop
                })
            
            # Add frame counter for webcam
            if is_webcam:
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            yield frame, cattle_data
    
    except Exception as e:
        print(f"Error in detection: {str(e)}")
    
    finally:
        cap.release()