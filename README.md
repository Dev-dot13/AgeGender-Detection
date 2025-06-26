# Age and Gender Detection using Computer Vision

## Project Description
Real-time age and gender detection system using VGG16-based multi-task learning.

## Features
- Age prediction in ranges (0-12, 13-18, etc.)
- Gender prediction (Male/Female)
- Real-time webcam detection

## Requirements
- Python 3.8+
- TensorFlow 2.x with GPU support
- OpenCV
- See requirements.txt for full list

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python src/training.py` 
3. Run real-time detection: `python src/realtime_detection.py`
