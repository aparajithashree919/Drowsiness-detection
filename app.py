from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
from functools import wraps
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

app = Flask(__name__)
app.secret_key = '060919'

# Dummy credentials
users = {"admin": "pass123"}

# Global alert state
alert_state = {"alert": "", "status": "Normal", "yawns": 0}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(alert_state)

# ------------- DROWSINESS & YAWN DETECTION LOGIC -------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 14
YAWN_THRESH = 20
COUNTER = 0
YAWN_COUNTER = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

camera = cv2.VideoCapture(0)

def gen_frames():
    global COUNTER, YAWN_COUNTER, alert_state
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray)

        current_alert = ""
        current_status = "Normal"

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            lip = lip_distance(shape)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    current_alert = "DROWSINESS ALERT!"
                    current_status = "Drowsy"
                    cv2.putText(frame, current_alert, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                COUNTER = 0

            if lip > YAWN_THRESH:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= 15:
                    current_alert = "YAWNING ALERT!"
                    current_status = "Yawning"
                    alert_state["yawns"] += 1
                    cv2.putText(frame, current_alert, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                YAWN_COUNTER = 0

        alert_state["alert"] = current_alert
        alert_state["status"] = current_status if current_alert else "Normal"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)