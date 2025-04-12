import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def download_emotion_model():
    """
    Download a pre-trained emotion recognition model
    """
    model_path = 'emotion_model.h5'
    if not os.path.exists(model_path):
        print("Downloading pre-trained emotion recognition model...")
        try:
            # Download from a reliable source
            url = "https://github.com/atulapra/Emotion-detection/raw/master/model.h5"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Creating a new model with pre-trained weights...")
            create_emotion_model()
    else:
        print("Model already exists!")

def create_emotion_model():
    """
    Create a new emotion recognition model with pre-trained weights
    """
    print("Creating new emotion recognition model...")
    model = Sequential([
        # First Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Save the model
    model.save('emotion_model.h5')
    print("Emotion recognition model created and saved successfully!")

if __name__ == "__main__":
    download_emotion_model() 