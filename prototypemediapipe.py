import tensorflow as tf
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import mediapipe as mp
import dlib
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LambdaCallback, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

label_encoder = None

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

nimgs = 150  # Increase the number of images to 150 for better accuracy
unknown_threshold = 0.9  # Set a threshold for unknown faces

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("static/predictor/shape_predictor_68_face_landmarks.dat")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load the trained model globally
model_path = 'static/mediapipe_recognition_model.keras'
label_encoder_path = 'static/label_encoder.pkl'  # Path for the label encoder
model = None
label_encoder = None  # Initialize label_encoder

print(f"Checking for model at: {model_path}")
if os.path.exists(model_path):
    print("Model path exists.")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        
        # Load the label encoder
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            print("Label encoder loaded successfully.")
        else:
            print("No label encoder found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("No trained model found. Please train the model first.")

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('static/faces/MediaPipealgo'):
    os.makedirs('static/faces/MediaPipealgo')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces/MediaPipealgo'))

def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    face_points = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_points.append((x, y, w, h))
    return face_points

def align_face(image):
    faces = detector(image, 1)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(image, face)
        aligned_face = dlib.get_face_chip(image, landmarks)
        return aligned_face
    return image

def train_cnn_model():
    global label_encoder
    faces, labels = [], []
    userlist = os.listdir('static/faces/MediaPipealgo')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/MediaPipealgo/{user}'):
            img = cv2.imread(f'static/faces/MediaPipealgo/{user}/{imgname}')
            if img is not None:
                aligned_face = align_face(img)
                face_points = extract_faces(aligned_face)
                for (x, y, w, h) in face_points:
                    face = aligned_face[y:y + h, x:x + w]
                    if face.size > 0:
                        face = cv2.resize(face, (50, 50))
                        faces.append(face)
                        labels.append(user)

    if not faces:
        print("No faces found.")
        return

    faces = np.array(faces).reshape(-1, 50, 50, 3)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(labels_encoded)

    X_train, X_val, y_train, y_val = train_test_split(faces, labels, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    model.save('static/mediapipe_recognition_model.keras')
    joblib.dump(label_encoder, 'static/label_encoder.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    if '_' not in name:  
        return  
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) in list(df['Roll']):
        df.loc[df['Roll'] == int(userid), 'Time'] = current_time
    else:
        new_row = pd.DataFrame([[username, int(userid), current_time]], columns=['Name', 'Roll', 'Time'])
        df = pd.concat([df, new_row])
    
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

def getallusers():
    userlist = os.listdir('static/faces/MediaPipealgo')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

# Ensure label_encoder is accessible in identify_face
def identify_face(face):
    global label_encoder
    model = tf.keras.models.load_model(model_path)
    if face is None or face.size == 0:
        print("No face detected or face array is empty.")
        return None

    try:
        facearray = cv2.resize(face, (50, 50))
        facearray = facearray.reshape(1, 50, 50, 3)
    except cv2.error as e:
        print(f"Error during resizing: {e}")
        return None

    if model is None:
        print("Model not loaded. Please train the model first.")
        return "Unknown"

    if label_encoder is None:
        print("Label encoder not initialized. Please train the model first.")
        return "Unknown"

    predictions = model.predict(facearray)
    max_prob = np.max(predictions)
    predicted_label = np.argmax(predictions)

    # Debugging output
    print(f"Predicted label index: {predicted_label}, Probability: {max_prob}")  
    print(f"Label Encoder Classes: {label_encoder.classes_}")

    if max_prob < unknown_threshold:
        return "Unknown"
    
    predicted_class = label_encoder.inverse_transform([predicted_label])
    print(f"Predicted class: {predicted_class[0]}, Max Probability: {max_prob}")
    return predicted_class[0] if len(predicted_class) > 0 else "Unknown"

@app.route('/')
def home():
    session['progress'] = 0
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, algo='MediaPipe', facereco = False)

@app.route('/start', methods=['GET'])
def start():
    
    names, rolls, times, l = extract_attendance()

    if 'mediapipe_recognition_model.keras' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    cv2.namedWindow('Attendance', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        face_points = extract_faces(frame)
        for (x, y, w, h) in face_points:
            if w > 0 and h > 0:  # Check if width and height are valid
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)

                face = frame[y:y + h, x:x + w]
                identified_person = identify_face(face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x, y), (50, 50, 255), -1)
                cv2.putText(frame, identified_person, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                if identified_person is not None:
                    add_attendance(identified_person)
                else:
                    print("No person identified")

        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return redirect('/')

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/MediaPipealgo/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        if not ret:
            break

        face_points = extract_faces(frame)
        for (x, y, w, h) in face_points:
            if w > 0 and h > 0:  # Ensure the bounding box dimensions are valid
                face_region = frame[y:y + h, x:x + w]
                if face_region.size > 0:  # Ensure the face region is not empty
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, 'Collecting Images: ' + str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.imshow('Adding Faces', frame)
                    if i < nimgs:
                        imgname = os.path.join(userimagefolder, f'image_{i + 1}.jpg')
                        cv2.imwrite(imgname, face_region)  # Save the face region
                        i += 1

        if cv2.waitKey(1) == 27 or i == nimgs:
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/train', methods=['GET'])
def train():
    session['progress'] = 0
    train_cnn_model()
    return jsonify({'status': 'Model trained successfully'})

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify({'progress': session.get('progress', 0)})

# Add this route to display users
@app.route('/users/', methods=['GET'])
def display_users():
    userlist = os.listdir('static/faces/MediaPipealgo')
    return jsonify(userlist)  # Return the list of users as a JSON response

# Add this route to remove a user
@app.route('/remove_user/<username>', methods=['POST'])
def remove_user(username):
    user_folder = os.path.join('static/faces/MediaPipealgo', username)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)  # Remove the directory
        return jsonify({'message': f'Successfully removed {username}'}), 200
    return jsonify({'message': 'User  not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)