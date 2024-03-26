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
    username = request.form['username']
    image = request.files['image']
    nparr = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    is_face = face_exists(image)
    if not is_face:
        return {'status': False, 'msg': 'Face not found.'}
    encodings = face_encodings(image, is_face)
    data = {'username': username, 'encodings': encodings}
    data_file = 'data.pkl'
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)
        for dt in existing_data:
            if dt['username'] == username:
                return {'status': False, 'msg': 'Username already exists.'}
            existing_encodings = np.array(dt['encodings'])
            current_encodings = np.array(encodings)
            distances = np.linalg.norm(existing_encodings - current_encodings, axis=1)
            tolerance = 0.6
            if distances[0] <= tolerance:
                return {'status': False, 'msg': 'Data already exists.'}
        existing_data.append(data)
        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        with open(data_file, 'wb') as f:
            pickle.dump([data], f)
    return {'status': True, 'msg': 'Member added successfully.'}

def face_exists(image):
    global face_locs
    face_locs = fcr.face_locations(image)
    return face_locs

def face_encodings(image, face):
    encodings = fcr.face_encodings(image, face)
    return encodings

def generate_frames():
    '''generate camera frames'''
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        thread = threading.Thread(target=face_exists, args=(frame,))
        if not thread.is_alive():
            thread.start()
            if face_locs:
                for top, right, bottom, left in face_locs:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)