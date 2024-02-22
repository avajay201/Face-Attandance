'''needed packages'''
from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
from datetime import datetime
import numpy as np
import mediapipe as mp
import os
import json


mdp_face = mp.solutions.face_mesh # load face mesh
members_data_file_path = 'members-data.json' # members data file
attendance_data_file_path = 'attendance-data.json' # attendance data file

app = Flask(__name__) # initialize flask app
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # set secret key in app


@app.route('/')
def index():
    '''home page'''
    create_attendance() # create attandance file if not exist
    msg = session.pop('msg', '') # remove stored message
    return render_template('index.html', msg=msg)


@app.route('/add-member', methods=['POST'])
def add_member():
    '''add member'''
    member_name = request.form['member_name']
    member_image = request.files['member_image']

    if not member_name or not member_image:
        return {'status': False, 'msg': 'Data not found.'}

    try:
        image_path = os.path.join('media', member_image.filename)
        member_image.save(image_path) # save member image
        face_exist = fetch_face(image_path) # check face exists or not in image
        if face_exist is not None:
            data = {
                'member_name': member_name,
                'image_path': image_path
            }
            save_to_json(members_data_file_path, data) # save member data in json file
            create_attendance(member_name) # create attandance file if not exist
            session['msg'] = 'Member added successfully!'
            return redirect(url_for('index'))
        os.remove(image_path)
        session['msg'] = 'Please enter valid image.'
        return redirect(url_for('index'))
    except Exception as e:
        print('Error:', e)
        session['msg'] = 'Failed to add member.'
        os.remove(image_path)
        return redirect(url_for('index'))


def save_to_json(file_path, data, update_attendance=False):
    '''save data into json file'''
    if os.path.exists(file_path):
        if update_attendance:
            # update member attendance
            updated_data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            for i, _ in enumerate(json_data):
                if json_data[i]['Name'] == data['Name']:
                    json_data[i]['Status'] = data['Status']
                    json_data[i]['Time'] = data['Time']
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, indent=4)
            return
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        json_data.append(data)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=4)
    else:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([data], file, indent=4)


def generate_frames():
    '''generate camera images'''
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    while True:
        ret, frame = cap.read()
        cv2.imwrite('static/live.jpg', frame)
        if not ret:
            break
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/camera_feed')
def camera_feed():
    '''use camera image in frontend'''
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/verify_name', methods=['POST'])
def verify_name():
    '''verify entered name exist or not in stored data'''
    name = request.form['name']
    found = False
    if os.path.exists(members_data_file_path):
        with open(members_data_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        for fd in json_data:
            if fd['member_name'] == name:
                found = True
    if not found:
        return {'status': False, 'msg': 'Member not found.'}

    # check already verified or not
    if os.path.exists(attendance_data_file_path):
        with open(attendance_data_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        for dt in json_data:
            if dt['Name'] == name and dt['Status'] == 'Present':
                return {'status': False, 'msg': 'Already verified.'}

    return {'status': True}


@app.route('/verify_member', methods=['POST'])
def verify_member():
    '''verify member'''
    name = request.form['name'] # member name
    # check member name exists or not
    if os.path.exists(members_data_file_path):
        with open(members_data_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        for fd in json_data:
            if fd['member_name'] == name:
                member_image = fd['image_path']
        face_exist = verify_face(member_image)
        if face_exist is not None:
            data = {
                'Name': name,
                'Time': datetime.now().strftime("%I:%M:%S %p, %d-%m-%Y"),
                'Status': 'Present'
            }
            save_to_json(attendance_data_file_path, data, True) # save data in json file
            return {'status': True}
    return {'status': False}


def verify_face(stored_face_path):
    '''verify live image to stored data'''
    face_match_path = 'static/live.jpg' # live camera image

    face_region = fetch_face(face_match_path)
    if face_region is not None:
        face_name = compare_with_known_faces(stored_face_path, face_region)
        return face_name


def fetch_face(image):
    '''fetch face boundry and face reason'''
    face_boundry = []
    face_region = None
    try:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        face_process = mdp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        results = face_process.process(image)
        if results.multi_face_landmarks:
            face_landmark = results.multi_face_landmarks[0]
            face_border_landmarks = list(mdp_face.FACEMESH_FACE_OVAL)
            # collecting face border landmarks
            for _, (x, y) in enumerate(face_border_landmarks):
                h, w, c = image.shape
                x, y = int(face_landmark.landmark[x].x * w), int(face_landmark.landmark[x].y * h)
                face_boundry.append((x, y))

            # Calculate bounding box of the face region
            xmin = min(x for x, y in face_boundry)
            xmax = max(x for x, y in face_boundry)
            ymin = min(y for x, y in face_boundry)
            ymax = max(y for x, y in face_boundry)
            width = xmax - xmin
            height = ymax - ymin
            # Extract the face region from the image
            face_region = image[ymin:ymin + height, xmin:xmin + width]
        return face_region
    except Exception as err:
        print('Error while detecting face:', err)
        return face_region


def compare_with_known_faces(stored_data, face_region):
    '''compare the face region with known faces using Mean Squared Error (MSE)'''
    try:
        mse_values = []
        known_face_image_path = stored_data
        known_face_image = cv2.imread(known_face_image_path)
        mse = calculate_mse(face_region, known_face_image)
        mse_values.append(mse)
        if mse_values:
            min_mse_index = np.argmin(mse_values)
            min_mse_value = mse_values[min_mse_index]
            some_threshold = 28000  # adjust this value based on experimentation
            if min_mse_value < some_threshold:
                return True
    except Exception as err:
        print('Error while comparing face:', err)
        return None

def calculate_mse(image1, image2):
    '''Calculate Mean Squared Error (MSE) between two images'''
    try:
        # Convert images to float32
        image1 = image1.astype(np.float32)

        # Resize image2 to match the shape of image1
        image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        image2_resized = image2_resized.astype(np.float32)

        mse = np.sum((image1 - image2_resized) ** 2) / float(image1.shape[0] * image1.shape[1])
        return mse
    except Exception as err:
        print('Error while calculating MSE:', err)
        return None


def create_attendance(add_member=False):
    '''create attendances'''
    if (os.path.exists(members_data_file_path) and not os.path.exists(attendance_data_file_path)) or add_member:
        if add_member:
            # add new member attendance
            data = {
                'Name': add_member,
                'Time': datetime.now().strftime("%I:%M:%S %p, %d-%m-%Y"),
                'Status': 'Absent'
            }
            json_data = [data]
            if os.path.exists(attendance_data_file_path):
                with open(attendance_data_file_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                json_data.append(data)
            with open(attendance_data_file_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, indent=4)
            return

        with open(members_data_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        members = [member['member_name'] for member in json_data]
        for name in members:
            data = {
                'Name': name,
                'Time': datetime.now().strftime("%I:%M:%S %p, %d-%m-%Y"),
                'Status': 'Absent'
            }
            if os.path.exists(attendance_data_file_path):
                with open(attendance_data_file_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                json_data.append(data)
                with open(attendance_data_file_path, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file, indent=4)
            else:
                with open(attendance_data_file_path, 'w', encoding='utf-8') as file:
                    json.dump([data], file, indent=4)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # run app
