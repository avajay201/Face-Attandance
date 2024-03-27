'''needed packages'''
from flask import Flask, render_template, request, Response
import cv2
import face_recognition as fcr
import numpy as np
import pickle
import os
import threading


app = Flask(__name__)
face_locs = None
data_file = 'data.pkl'

@app.route('/')
def home():
    '''home page'''
    return render_template('home.html')

@app.route('/camera-feed')
def camera_feed():
    '''show live camera prev'''
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add-member', methods=['POST'])
def add_member():
    '''add member's data'''
    username = request.form['username']
    image = request.files['image']
    nparr = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    is_face = face_exists(image)
    if not is_face:
        return {'status': False, 'msg': 'Face not found.'}
    encodings = face_encodings(image, is_face)
    data = {'username': username, 'encodings': encodings}
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)
        for dt in existing_data:
            if dt['username'] == username:
                return {'status': False, 'msg': 'Username already exists.'}
            matched = face_match(dt['encodings'], encodings)
            if matched:
                return {'status': False, 'msg': 'Data already exists.'}
        existing_data.append(data)
        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        with open(data_file, 'wb') as f:
            pickle.dump([data], f)
    return {'status': True, 'msg': 'Member added successfully.'}

def face_exists(image):
    '''face locations'''
    global face_locs
    face_locs = fcr.face_locations(image)
    return face_locs

def face_encodings(image, face):
    '''face encodings'''
    encodings = fcr.face_encodings(image, face)
    return encodings

def verify_member(encodings):
    '''verify member'''
    if not os.path.exists(data_file):
        return
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    for dt in data:
        matched = face_match(dt['encodings'], encodings)
        if matched:
            return dt['username']

def face_match(encodings1, encodings2):
    '''face match'''
    existing_encodings = np.array(encodings1)
    current_encodings = np.array(encodings2)
    distances = np.linalg.norm(existing_encodings - current_encodings, axis=1)
    tolerance = 0.5
    if distances[0] <= tolerance:
        return True

def generate_frames():
    '''generate camera frames'''
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        face_locs_thread = threading.Thread(target=face_exists, args=(frame,))
        if not face_locs_thread.is_alive():
            face_locs_thread.start()
            if face_locs:
                encodings = face_encodings(frame, face_locs)
                verified = verify_member(encodings)
                if verified:
                    cv2.putText(frame, verified, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                for top, right, bottom, left in face_locs:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)