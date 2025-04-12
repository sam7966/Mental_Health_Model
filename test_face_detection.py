import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_emotion_model():
    """
    Load the pre-trained emotion recognition model
    """
    try:
        model = load_model('emotion_model.h5')
        return model
    except:
        print("Error: Could not load emotion model. Please make sure emotion_model.h5 exists.")
        return None

def preprocess_face(face_img):
    """
    Preprocess face image for emotion recognition
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Resize to 48x48 (model input size)
    gray = cv2.resize(gray, (48, 48))
    
    # Convert to array and normalize
    face_array = img_to_array(gray)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = face_array / 255.0
    
    return face_array

def test_webcam_face_detection():
    """
    Test face detection and emotion recognition with webcam input
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load emotion recognition model
    emotion_model = load_emotion_model()
    if emotion_model is None:
        return
    
    print("Starting webcam face detection and emotion recognition...")
    print("Press 'q' to quit")
    
    # Set minimum confidence threshold for emotion prediction
    confidence_threshold = 0.3
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),  # Increased minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process and display detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
                
            face_img = cv2.resize(face_img, (200, 200))
            
            # Preprocess face for emotion recognition
            processed_face = preprocess_face(face_img)
            
            # Predict emotion
            predictions = emotion_model.predict(processed_face)
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            
            # Only display if confidence is above threshold
            if confidence > confidence_threshold:
                emotion = EMOTIONS[emotion_idx]
                # Display emotion and confidence
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Display the face in a separate window
                cv2.imshow('Detected Face', face_img)
            else:
                # Display "Unknown" if confidence is too low
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the original frame with face detection and emotion
        cv2.imshow('Webcam', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam_face_detection() 