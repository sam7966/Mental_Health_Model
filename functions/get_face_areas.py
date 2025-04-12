import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import torch

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoCamera(object):
    def __init__(self, source=0, conf=0.7, path_video=None):  # source=0 for default webcam
        self.source = source
        self.conf = conf
        self.path_video = path_video
        self.cur_frame = 0
        self.video = None
        self.dict_face_area = {}
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load emotion recognition model
        self.emotion_model = self.load_emotion_model()

    def load_emotion_model(self):
        try:
            # Try to load existing models
            model_name = '0_66_49_wo_gl'
            tf_model_path = f'models_EmoAffectnet/weights_{model_name}.h5'
            torch_model_path = f'models_EmoAffectnet/torchscript_model_{model_name}.pth'
            
            if os.path.exists(tf_model_path):
                print(f"Loading TensorFlow model from {tf_model_path}")
                model = load_model(tf_model_path)
            elif os.path.exists(torch_model_path):
                print(f"Loading PyTorch model from {torch_model_path}")
                model = torch.jit.load(torch_model_path)
                model.eval()
            else:
                print("No pre-trained models found. Creating new emotion recognition model...")
                return self.create_emotion_model()
                
            print("Emotion recognition model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new emotion recognition model...")
            return self.create_emotion_model()

    def create_emotion_model(self):
        # Load MobileNetV2 as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers for emotion recognition
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(7, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Save the model
        model.save('emotion_model_mobilenet.h5')
        print("New emotion recognition model created and saved")
        return model

    def __del__(self):
        if self.video is not None:
            self.video.release()
        
    def preprocess_image(self, cur_fr):
        cur_fr = utils.preprocess_input(cur_fr, version=2)
        return cur_fr
        
    def channel_frame_normalization(self, cur_fr):
        cur_fr = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
        cur_fr = cv2.resize(cur_fr, (224, 224), interpolation=cv2.INTER_AREA)
        cur_fr = img_to_array(cur_fr)
        cur_fr = self.preprocess_image(cur_fr)
        return cur_fr

    def preprocess_face_for_emotion(self, face_img):
        # Resize to 224x224 (required by MobileNetV2)
        face_img = cv2.resize(face_img, (224, 224))
        # Convert to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
            
    def get_frame(self):
        print("\nStarting video feed...")
        
        # Initialize video capture
        if self.path_video is not None:
            if not os.path.exists(self.path_video):
                print(f"Error: File not found: {self.path_video}")
                return None, 0
            self.video = cv2.VideoCapture(self.path_video)
            print(f"Processing file: {self.path_video}")
        else:
            self.video = cv2.VideoCapture(self.source)
            print("Starting webcam feed...")
        
        # Check if video opened successfully
        if not self.video.isOpened():
            print("Error: Could not open video source")
            return None, 0
            
        print('Press "q" to quit')
        
        # Create windows for display
        cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detected Face', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            self.cur_frame += 1
            name_img = str(self.cur_frame).zfill(6)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and process face
                cur_fr = frame[y:y+h, x:x+w]
                if cur_fr.size > 0:  # Check if face area is valid
                    self.dict_face_area[name_img] = self.channel_frame_normalization(cur_fr)
                    
                    # Process face for emotion recognition
                    emotion_input = self.preprocess_face_for_emotion(cur_fr)
                    if isinstance(self.emotion_model, torch.jit.ScriptModule):
                        # Handle PyTorch model
                        with torch.no_grad():
                            emotion_input = torch.from_numpy(emotion_input).permute(0, 3, 1, 2)
                            emotion_pred = self.emotion_model(emotion_input).numpy()[0]
                    else:
                        # Handle TensorFlow model
                        emotion_pred = self.emotion_model.predict(emotion_input)[0]
                        
                    emotion_idx = np.argmax(emotion_pred)
                    emotion = EMOTIONS[emotion_idx]
                    confidence = emotion_pred[emotion_idx]
                    
                    # Display emotion and confidence
                    emotion_text = f"{emotion}: {confidence:.2f}"
                    cv2.putText(display_frame, emotion_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Display face in separate window
                    face_display = cv2.resize(cur_fr, (224, 224))
                    cv2.imshow('Detected Face', face_display)
            
            # Display the frame with face detection
            cv2.imshow('Video Feed', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        cv2.destroyAllWindows()
        return self.dict_face_area, self.cur_frame

def main():
    if len(sys.argv) > 1:
        # If video path is provided, use it
        video_path = sys.argv[1]
        camera = VideoCamera(path_video=video_path)
    else:
        # Otherwise use webcam
        camera = VideoCamera(source=0)
        
    face_areas, total_frames = camera.get_frame()
    
    if face_areas is not None:
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {total_frames}")
        print(f"Faces detected: {len(face_areas)}")
        if len(face_areas) > 0:
            print("Sample face dimensions:", face_areas[next(iter(face_areas))].shape)
    else:
        print("No faces were detected in the video.")

if __name__ == "__main__":
    main()