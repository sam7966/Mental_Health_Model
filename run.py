import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

import warnings
import sys
import subprocess
import cv2
from functions.get_face_areas import VideoCamera

warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default='models/EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default='models/LSTM/RAVDESS_with_config.h5',
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'opencv-python',
        'numpy',
        'tensorflow',
        'torch'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("All required packages installed successfully!")

def run_face_detection():
    """Run face detection on video or webcam"""
    # Check dependencies
    check_dependencies()
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if a video file is provided as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        print(f"Processing video file: {video_path}")
        camera = VideoCamera(path_video=video_path)
    else:
        # Check for GIF files in the gif directory
        gif_dir = os.path.join(current_dir, 'gif')
        if os.path.exists(gif_dir):
            gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
            if gif_files:
                gif_path = os.path.join(gif_dir, gif_files[0])
                print(f"Using GIF file: {gif_path}")
                camera = VideoCamera(path_video=gif_path)
            else:
                print("No GIF files found. Using webcam...")
                camera = VideoCamera(source=0)
        else:
            print("No GIF directory found. Using webcam...")
            camera = VideoCamera(source=0)
    
    # Run face detection
    face_areas, total_frames = camera.get_frame()
    
    if face_areas is not None:
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {total_frames}")
        print(f"Faces detected: {len(face_areas)}")
        if len(face_areas) > 0:
            print("Sample face dimensions:", face_areas[next(iter(face_areas))].shape)
    else:
        print("No faces were detected in the video.")

def pred_one_video(path):
    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frame()
    name_frames = list(dict_face_areas.keys())
    face_areas = list(dict_face_areas.values())
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    features = EE_model(np.stack(face_areas))
    seq_paths, seq_features = sequences.sequences(name_frames, features)
    pred = LSTM_model(np.stack(seq_features)).numpy()
    all_pred = []
    all_path = []
    for id, c_p in enumerate(seq_paths):
        c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1])+1)]
        c_pr = [pred[id]]*len(c_f)
        all_pred.extend(c_pr)
        all_path.extend(c_f)    
    m_f = [str(i).zfill(6) for i in range(int(all_path[-1])+1, total_frame+1)] 
    m_p = [all_pred[-1]]*len(m_f)
    
    df=pd.DataFrame(data=all_pred+m_p, columns=label_model)
    df['frame'] = all_path+m_f
    df = df[['frame']+ label_model]
    df = sequences.df_group(df, label_model)
    
    if not os.path.exists(args.path_save):
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    df.to_csv(os.path.join(args.path_save,filename), index=False)
    end_time = time.time() - start_time
    mode = stats.mode(np.argmax(pred, axis=1))[0]
    print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_model[mode])
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in enumerate(path_all_videos):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    run_face_detection()
    input("Press Enter to exit...")