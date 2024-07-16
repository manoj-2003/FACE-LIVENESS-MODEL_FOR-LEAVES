from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import face_recognition
import os
import pymongo
from datetime import datetime
import base64

app = Flask(__name__)

# Load the trained liveness detection model
model_path = '/home/rgukt/Desktop/FACE PROJECT/face_liveness_detection/model/liveness_model.h5'
model = load_model(model_path)

# Define the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Path to the directory with known face images
known_faces_dir = '/home/rgukt/Desktop/FACE PROJECT/face_liveness_detection/images'

# Encode known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(known_faces_dir, filename)
        img = face_recognition.load_image_file(img_path)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use the filename without extension as the name

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["face_liveness_detection"]
collection = db["captures"]

def generate_frames():
    cap = cv2.VideoCapture(1)  # 0 for the default camera, update if using a different camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert the frame to RGB (for face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face locations and encodings for face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face encodings with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (x, y, w, h), name in zip(faces, face_names):
            # Extract the face region
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (96, 96))  # Resize to match the input size of the model

            # Preprocess the face image for prediction
            img_array = image.img_to_array(face_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image data

            # Perform prediction using the model
            prediction = model.predict(img_array)
            print(f'Prediction: {prediction}')  # Debugging line to check prediction values

            # Interpret the prediction
            threshold = 0.5  # Default threshold for binary classification
            if prediction[0][0] > threshold:
                result = 'FAKE'
            else:
                result = 'REAL'

            # Draw bounding box and display prediction result
            color = (0, 0, 255) if result == 'FAKE' else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{name}: {result}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Capture the face image and timestamp
            face_image = frame[y:y + h, x:x + w]
            _, buffer = cv2.imencode('.jpg', face_image)
            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Store the data in MongoDB
            document = {
                "name": name,
                "result": result,
                "timestamp": timestamp,
                "image": face_image_base64
            }
            collection.insert_one(document)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)
