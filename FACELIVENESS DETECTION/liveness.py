import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model_path = '/home/rgukt/Desktop/FACE PROJECT/face_liveness_detection/model/liveness_model.h5'
model = load_model(model_path)

# Define the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, update if using a different camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
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
        threshold = 0.3  # Default threshold for binary classification
        if prediction[0][0] < threshold:
            result = 'FAKE'
        else:
            result = 'REAL'

        # Draw bounding box and display prediction result
        color = (0, 0, 255) if result == 'FAKE' else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Face Liveness Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
