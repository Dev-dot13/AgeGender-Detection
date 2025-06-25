import cv2
import numpy as np
from tensorflow.keras.models import load_model

class AgeGenderDetector:
    def __init__(self, model_path, age_classes, gender_classes):
        self.model = load_model(model_path)
        self.age_classes = age_classes
        self.gender_classes = gender_classes
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_face(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        return np.expand_dims(face_img, axis=0)
    
    def predict_age_gender(self, face_img):
        processed = self.preprocess_face(face_img)
        age_pred, gender_pred = self.model.predict(processed)
        
        age = np.argmax(age_pred[0])
        gender = "Male" if np.argmax(gender_pred[0]) == 1 else "Female"
        
        return age, gender
    
    def run_realtime_detection(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    age, gender = self.predict_age_gender(face_img)
                    
                    # Map age index to actual range
                    age_ranges = ["0-12", "13-18", "19-25", "26-35", "36-50", "51-65", "66+"]
                    age_label = age_ranges[age] if age < len(age_ranges) else str(age)
                    
                    # Display results
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Age: {age_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Gender: {gender}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                except Exception as e:
                    print(f"Error in prediction: {e}")
                    continue
                    
            cv2.imshow('Age and Gender Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AgeGenderDetector('models/vgg16_age_gender.h5', age_classes=9, gender_classes=2)
    detector.run_realtime_detection()