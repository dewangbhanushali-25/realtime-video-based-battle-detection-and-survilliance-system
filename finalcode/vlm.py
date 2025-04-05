import os
import time
from PIL import Image
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize API client if key is available
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

# Cache for storing descriptions to avoid duplicate API calls
activity_cache = {}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests

def describe_with_gemini(image_path, retry_attempts=2):
    """
  make a request to the Gemini API to describe the activity of cattle in the image.
  and makes 2 attempts

    """
    global last_request_time
    
    if not API_KEY:
        return "API not configured. Set GEMINI_API_KEY environment variable."
    
   
    if not os.path.exists(image_path):
        return "Image file not found"
    
    # Check cache to avoid duplicate API calls
    if image_path in activity_cache:
        return activity_cache[image_path]
    
    # Rate limiting
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - time_since_last_request)
    
    # Initialize Gemini model
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        return f"Failed to initialize Gemini model: {str(e)[:100]}"
    
    # Open and process image
    try:
        image = Image.open(image_path)
        
        # Resize if too large (Gemini has input size limits)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
    except Exception as e:
        return f"Error processing image: {str(e)[:100]}"
    
    # Run inference with retries
    attempt = 0
    while attempt <= retry_attempts:
        try:
            # Update last request time
            last_request_time = time.time()
            
            # Define a more specific prompt for better results
            prompt = """
            Analyze this image of cattle/livestock and describe:
            1. What type of animal is it (cow, sheep, buffalo, goat, pig)?
            2. What is the animal doing (eating, walking, resting, etc.)?
            3. What is its posture or position?
            
            
            """
            
            # Generate response
            response = model.generate_content([prompt, image])
            description = response.text.strip()
            
            # Cache the result
            activity_cache[image_path] = description
            
            return description
            
        except Exception as e:
            attempt += 1
            if attempt > retry_attempts:
                return f"Error after {retry_attempts} attempts: {str(e)[:100]}"
            time.sleep(1)  # Wait before retry
    
    return "Failed to get description"

def test_gemini_connection():
    """Test if the Gemini API is properly configured"""
    if not API_KEY:
        return False, "No API key configured"
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, test connection")
        return True, "Connection successful"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"