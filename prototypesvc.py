import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
import shutil

from keras.callbacks import LambdaCallback

app = Flask(__name__)

nimgs = 150  # Increase the number of images to 30 for better accuracy
unknown_threshold = 0.5  # Set a threshold for unknown faces

imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('static/faces/SVCalgo'):
    os.makedirs('static/faces/SVCalgo')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces/SVCalgo'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/svc_recognition_model.pkl')
    probabilities = model.predict_proba(facearray)
    predictions = model.predict(facearray)
    
    print("Probabilities:", probabilities)
    print("Predictions:", predictions)
    
    # Sort probabilities to find the two highest
    sorted_probs = np.sort(probabilities[0])[::-1]
    max_prob = sorted_probs[0]
    second_max_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
    
    # Check if the highest probability is much greater than the second highest
    confidence_margin = 0.2  # Adjust as needed
    if max_prob - second_max_prob < confidence_margin or max_prob < unknown_threshold:
        return "Unknown"
    
    # Check if the highest probability is above a certain threshold
    if max_prob < 0.7:  # Adjust this value as needed
        return "Unknown"
    
    return predictions[0]

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces/SVCalgo')
    
    total_images = 0
    for user in userlist:
        total_images += len(os.listdir(f'static/faces/SVCalgo/{user}'))
    
    print(f"Training started... Total images: {total_images}")
    
    processed_images = 0
    for user in userlist:
        name = user.split('_')[0]  # Extract the person's name from the folder name
        for imgname in os.listdir(f'static/faces/SVCalgo/{user}'):
            img = cv2.imread(f'static/faces/SVCalgo/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(name + '_' + user.split('_')[1])  # Use the person's name and id as the label
            
            processed_images += 1
            print(f"Processing image {processed_images}/{total_images}")
    
    faces = np.array(faces)

    # Update progress at the end of each epoch
    def update_progress(epoch, logs):
        session['progress'] = (epoch + 1) / 10 * 100

    print_callback = LambdaCallback(on_epoch_end=update_progress)
    
    # Using Support Vector Classifier (SVC)
    svc = SVC(kernel='rbf', gamma='scale', class_weight='balanced', probability=True)
    svc.fit(faces, labels, callbacks=[print_callback])
    joblib.dump(svc, 'static/svc_recognition_model.pkl')
    
    print("Model training completed.")

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    if '_' not in name:  # Check if the name contains an underscore
        return  # Exit the function if the format is incorrect
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
    userlist = os.listdir('static/faces/SVCalgo')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, algo='SVC', facereco = False)

identified_users = {}  # Format: {'person_name': (first_detection_time, last_detection_time, duration)}

# Adjust the 'start' function to track detection duration under one label
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'svc_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    cv2.namedWindow('Attendance', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    detection_timeout = 1  # Time in seconds to wait before resetting detection
    min_detection_duration = 2  # Minimum duration in seconds to count as a valid detection

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        faces_detected = extract_faces(frame)
        current_time = datetime.now()

        detected_now = []  # List to track currently detected users in this frame

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)

            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            detected_now.append(identified_person)

            if identified_person not in identified_users:
                # First detection of this person
                identified_users[identified_person] = (current_time, current_time, 0)
            else:
                # Update the last detection time
                first_detection_time, last_detection_time, duration = identified_users[identified_person]
                if (current_time - last_detection_time).total_seconds() <= detection_timeout:
                    # Update the last detection time and duration
                    duration += (current_time - last_detection_time).total_seconds()
                    identified_users[identified_person] = (first_detection_time, current_time, duration)
                else:
                    # Person was lost for too long, reset the detection
                    identified_users.pop(identified_person)
                    continue

                # Check if they've been detected for the required duration
                if duration >= min_detection_duration:
                    add_attendance(identified_person)
                    identified_users.pop(identified_person)  # Reset after marking attendance

            cv2.putText(frame, f'{identified_person}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Remove people who haven't been detected in the last frame
        users_to_remove = [user for user in identified_users if user not in detected_now]
        for user in users_to_remove:
            identified_users.pop(user)

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
    userimagefolder = 'static/faces/SVCalgo/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()  # Read the frame from the webcam
        if not ret:
            break  # Break the loop if frame capture fails

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            if j % 5 == 0 and i < nimgs:  # Capture images until nimgs is reached
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1

        cv2.imshow('Add Faces', frame)  # Show the webcam feed
        if i == nimgs:  # Break if nimgs is reached
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Break on 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/train', methods=['GET'])
def train():
    userlist = os.listdir('static/faces/SVCalgo')  # Get list of users

    if not userlist:  # Check if there are no users
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2,
                               mess='Please add images first!')  # Show message

    print('Training model...')
    session['progress'] = 0
    train_model()
    names, rolls, times, l = extract_attendance()
    return redirect('/')

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify({'progress': session.get('progress', 0)})

# Add this route to display users
@app.route('/users/', methods=['GET'])
def display_users():
    userlist = os.listdir('static/faces/SVCalgo')
    return jsonify(userlist)  # Return the list of users as a JSON response

# Add this route to remove a user
@app.route('/remove_user/<username>', methods=['POST'])
def remove_user(username):
    user_folder = os.path.join('static/faces/SVCalgo', username)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)  # Remove the directory
        return jsonify({'message': f'Successfully removed {username}'}), 200
    return jsonify({'message': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)