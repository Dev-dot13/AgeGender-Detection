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

## Dataset
This project uses the UTKFace dataset from kaggle, a large-scale face dataset with annotations for age, gender, and ethnicity.
- File Naming Format: `[age]_[gender]_[race]_[date&time].jpg`

## Data Preprocessing
- Preprocessing
    - Images are resized to 224x224 (compatible with VGG16).
    - Pixel values are normalized to [0, 1] (divided by 255.0).
    - Age is binned into groups (e.g., 0-12, 13-18, etc.) for classification.
    - Gender is treated as binary (0=Male, 1=Female).

- Train/Validation/Test Split
    - 80% Training
    - 10% Validation
    - 10% Testing

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python src/training.py` 
3. Run real-time detection: `python src/realtime_detection.py`
