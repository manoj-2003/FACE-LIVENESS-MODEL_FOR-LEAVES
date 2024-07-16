from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, send_file
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import face_recognition
import os
from pymongo import MongoClient
from datetime import datetime
import base64
import csv

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
client = MongoClient('mongodb://localhost:27017/')
db = client.school  # Database name
students_collection = db.students  # Collection name
outings_collection = db.outings  # Collection for outings

def fetch_student_details(student_id):
    return students_collection.find_one({"id": student_id})

def insert_outing_details(student_details, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outing_record = {
        "name": student_details['name'],
        "id": student_details['id'],
        "year": student_details['year'],
        "branch": student_details['branch'],
        "reason": reason,
        "Outtimestamp": timestamp,
        "Intimestamp": None
    }
    outings_collection.insert_one(outing_record)

def update_intimestamp(student_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outings_collection.update_one(
        {"id": student_id, "Intimestamp": None},
        {"$set": {"Intimestamp": timestamp}}
    )

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    results = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (x, y, w, h), name in zip(faces, face_names):
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (96, 96))
        img_array = image.img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)
        threshold = 0.5
        result = 'REAL' if prediction[0][0] >= threshold else 'FAKE'

        if result == 'REAL' and name != "Unknown":
            student_details = fetch_student_details(name)
            if student_details:
                results.append({
                    "name": student_details['name'],
                    "year": student_details['year'],
                    "branch": student_details['branch'],
                    "id": student_details['id'],
                    "result": result
                })
                existing_outing = outings_collection.find_one({"id": student_details['id'], "Intimestamp": None})
                if existing_outing:
                    update_intimestamp(student_details['id'])
                else:
                    insert_outing_details(student_details, "Reason Not Provided")
            else:
                results.append({"name": name, "result": "NOT FOUND IN DATABASE"})
        else:
            results.append({"name": name, "result": "FAKE" if result == 'FAKE' else "NOT FOUND IN DATABASE"})
    return results

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data['image']
    image_decoded = base64.b64decode(image_data)
    np_img = np.frombuffer(image_decoded, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = detect_and_predict(frame)

    if results and results[0]['result'] == 'REAL':
        student_id = results[0]['id']
        student_details = fetch_student_details(student_id)
        if student_details:
            results[0]['message'] = 'Student details retrieved successfully'
        else:
            results[0]['message'] = 'Student details not found in the database'
    else:
        for result in results:
            result['message'] = 'Fake or unrecognized face detected'
    
    return jsonify(results)

@app.route('/fetch_by_id', methods=['POST'])
def fetch_by_id():
    data = request.json
    student_id = data['id']
    student_details = fetch_student_details(student_id)
    if student_details:
        result = {
            "name": student_details['name'],
            "year": student_details['year'],
            "branch": student_details['branch'],
            "id": student_details['id'],
            "result": 'REAL'
        }
    else:
        result = {"result": "Unknown"}
    return jsonify([result])

@app.route('/submit_reason', methods=['POST'])
def submit_reason():
    data = request.json
    student_id = data['id']
    reason = data['reason']
    student_details = fetch_student_details(student_id)
    if student_details:
        insert_outing_details(student_details, reason)
        return jsonify({"message": "Outing reason submitted successfully."})
    else:
        return jsonify({"message": "Error: Student details not found."})

@app.route('/student_details/<student_id>')
def student_details_page(student_id):
    student_details = fetch_student_details(student_id)
    out = False  # Default value for out
    in_time = False  # Default value for in_time

    if student_details:
        existing_outing = outings_collection.find_one({"id": student_id, "Intimestamp": None})
        out = not bool(existing_outing)
        in_time = bool(existing_outing)

        return render_template('inandout.html', details=student_details, out=out, in_time=in_time)
    else:
        return render_template('inandout.html', details=student_details, out=out, in_time=in_time)

@app.route('/security', methods=['POST'])
def security():
    status = request.form.get('status')
    student_id = request.form.get('id')
    name = request.form.get('name')
    branch = request.form.get('branch')
    year = request.form.get('year')
    reason = request.form.get('reason')
    timestamp = request.form.get('timestamp')

    if status == '1':  # Leaving
        outing_record = {
            "name": name,
            "id": student_id,
            "branch": branch,
            "year": year,
            "reason": reason,
            "Outtimestamp": timestamp,
            "Intimestamp": None
        }
        outings_collection.insert_one(outing_record)
    elif status == '2':  # Entering
        outings_collection.update_one(
            {"id": student_id, "Intimestamp": None},
            {"$set": {"Intimestamp": timestamp}}
        )
    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    outings = list(outings_collection.find())
    csv_file_path = "/home/rgukt/Desktop/FACE PROJECT/face_liveness_detection/JUNE17/static/outings.csv"

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "ID", "Year", "Branch", "Reason", "Outtimestamp", "Intimestamp"])

        for outing in outings:
            writer.writerow([
                outing.get("name"),
                outing.get("id"),
                outing.get("year"),
                outing.get("branch"),
                outing.get("reason"),
                outing.get("Outtimestamp"),
                outing.get("Intimestamp")
            ])

    return send_file(csv_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5435)


"""from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import face_recognition
import os
from pymongo import MongoClient
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
client = MongoClient('mongodb://localhost:27017/')
db = client.school  # Database name
students_collection = db.students  # Collection name
outings_collection = db.outings  # Collection for outings

def fetch_student_details(student_id):
    return students_collection.find_one({"id": student_id})

def insert_outing_details(name, student_details, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outing_record = {
        "name": name,
        "id": student_details['id'],
        "year": student_details['year'],
        "branch": student_details['branch'],
        "reason": reason,
        "Outtimestamp": timestamp,
        "Intimestamp": None
    }
    outings_collection.insert_one(outing_record)

def update_intimestamp(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outings_collection.update_one(
        {"name": name, "Intimestamp": None},
        {"$set": {"Intimestamp": timestamp}}
    )

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    results = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (x, y, w, h), name in zip(faces, face_names):
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (96, 96))
        img_array = image.img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)
        threshold = 0.5
        result = 'REAL' if prediction[0][0] >= threshold else 'FAKE'

        if result == 'REAL' and name != "Unknown":
            student_details = fetch_student_details(name)
            if student_details:
                results.append({
                    "name": student_details['name'],
                    "year": student_details['year'],
                    "branch": student_details['branch'],
                    "id": student_details['id'],
                    "result": result
                })
                existing_outing = outings_collection.find_one({"name": name, "Intimestamp": None})
                if existing_outing:
                    update_intimestamp(name)
                else:
                    insert_outing_details(name, student_details, "Reason Not Provided")
            else:
                results.append({"name": name, "result": "Unknown"})
    return results

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data['image']
    image_decoded = base64.b64decode(image_data)
    np_img = np.frombuffer(image_decoded, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = detect_and_predict(frame)

    if results and results[0]['result'] == 'REAL':
        student_id = results[0]['id']
        student_details = fetch_student_details(student_id)
        if student_details:
            students_collection.update_one(
                {"id": student_id},
                {"$set": {
                    "name": student_details['name'],
                    "year": student_details['year'],
                    "branch": student_details['branch']
                }}
            )
            results[0]['message'] = 'Student details updated successfully'
        else:
            results[0]['message'] = 'Student details not found in the database'
    else:
        for result in results:
            result['message'] = 'Fake or unrecognized face detected'
    
    return jsonify(results)

@app.route('/fetch_by_id', methods=['POST'])
def fetch_by_id():
    data = request.json
    student_id = data['id']
    student_details = fetch_student_details(student_id)
    if student_details:
        result = {
            "name": student_details['name'],
            "year": student_details['year'],
            "branch": student_details['branch'],
            "id": student_details['id'],
            "result": 'REAL'
        }
    else:
        result = {"result": "Unknown"}
    return jsonify([result])

@app.route('/submit_reason', methods=['POST'])
def submit_reason():
    data = request.json
    name = data['name']
    reason = data['reason']
    student_details = fetch_student_details(name)
    if student_details:
        existing_outing = outings_collection.find_one({"name": name, "Intimestamp": None})
        if existing_outing:
            outings_collection.update_one(
                {"name": name, "Intimestamp": None},
                {"$set": {"reason": reason}}
            )
        else:
            insert_outing_details(name, student_details, reason)
        return jsonify({"message": "Outing reason submitted successfully."})
    else:
        return jsonify({"message": "Error: Student details not found."})

@app.route('/student_details/<student_id>')
def student_details_page(student_id):
    student_details = fetch_student_details(student_id)
    out = False  # Default value for out
    in_time = False  # Default value for in_time

    if student_details:
        existing_outing = outings_collection.find_one({"id": student_id, "Intimestamp": None})
        out = not bool(existing_outing)
        in_time = bool(existing_outing)

        return render_template('inandout.html', details=student_details, out=out, in_time=in_time)
    else:
        return render_template('inandout.html', details=student_details, out=out, in_time=in_time)

@app.route('/security', methods=['POST'])
def security():
    status = request.form.get('status')
    student_id = request.form.get('id')
    name = request.form.get('name')
    branch = request.form.get('branch')
    year = request.form.get('year')
    reason = request.form.get('reason')  # Get the reason from the form
    timestamp = request.form.get('timestamp')

    if status == '1':  # Leaving
        student_details = fetch_student_details(student_id)
        if student_details:
            insert_outing_details(name, student_details, reason)
    elif status == '2':  # Entering
        existing_outing = outings_collection.find_one({"name": name, "Intimestamp": None})
        if existing_outing:
            # Update the reason in the existing outing record
            outings_collection.update_one(
                {"name": name, "Intimestamp": None},
                {"$set": {"reason": reason, "Intimestamp": timestamp}}
            )
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True,port=5653)"""
