import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_images(folder_path, target_size=(224, 224)):
    images = []
    age_labels = []
    gender_labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # Parse age and gender from filename (UTKFace format: age_gender_race_date.jpg)
            age = int(filename.split('_')[0])
            gender = int(filename.split('_')[1])
            
            # Read and preprocess image
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize
            
            images.append(img)
            age_labels.append(age)
            gender_labels.append(gender)
    
    return np.array(images), np.array(age_labels), np.array(gender_labels)

def prepare_datasets(data_path):
    # Load images
    X, y_age, y_gender = load_images(data_path)
    
    # Convert age to age groups (adjust ranges as needed)
    age_bins = [0, 12, 18, 25, 35, 50, 65, 100]
    y_age_group = np.digitize(y_age, age_bins)
    
    # Split data
    X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
        X, y_age_group, y_gender, test_size=0.2, random_state=42)
    
    X_val, X_test, y_age_val, y_age_test, y_gender_val, y_gender_test = train_test_split(
        X_test, y_age_test, y_gender_test, test_size=0.5, random_state=42)
    
    return (X_train, y_age_train, y_gender_train), (X_val, y_age_val, y_gender_val), (X_test, y_age_test, y_gender_test)