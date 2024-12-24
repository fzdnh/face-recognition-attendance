import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback

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

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

nimgs = 150  # Increase the number of images to 30 for better accuracy
unknown_threshold = 1  # Set a threshold for unknown faces

imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the model globally
model = tf.keras.models.load_model('static/cnn_recognition_model.h5')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('static/faces/CNNalgo'):
    os.makedirs('static/faces/CNNalgo')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces/CNNalgo'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    global model  # Use the globally loaded model
    
    facearray = np.array(facearray).astype('float32') / 255.0
    facearray = facearray.reshape(1, 50, 50, 3)
    
    predictions = model.predict(facearray)
    face_embedding = predictions[0]
    
    # Normalize the face embedding
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    
    print("Input Face Embedding:", face_embedding)  # Debugging output
    
    userlist = os.listdir('static/faces/CNNalgo')
    max_similarity = -1
    predicted_user = None
    
    for user in userlist:
        user_embedding = np.load(f'static/faces/CNNalgo/{user}/embedding.npy')
        
        # Normalize the user embedding
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        
        # Calculate the cosine similarity
        similarity = np.dot(face_embedding, user_embedding)
        
        print(f"Similarity to {user}: {similarity}")  # Debugging output
        print("User  Embedding:", user_embedding)  # Debugging output
        
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_user = user
    
    if max_similarity < unknown_threshold:  # Adjust this threshold as needed
        return "Unknown"
    
    return predicted_user

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces/CNNalgo')
    
    total_images = sum(len([img for img in os.listdir(f'static/faces/CNNalgo/{user}') if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]) for user in userlist)
    print(f"Training started... Total images: {total_images}")
    
    # Load images and labels
    processed_images = 0
    for user in userlist:
        for imgname in os.listdir(f'static/faces/CNNalgo/{user}'):
            if not imgname.endswith('.jpg') and not imgname.endswith('.jpeg') and not imgname.endswith('.png'):
                continue  # Skip .npy files
            
            img_path = f'static/faces/CNNalgo/{user}/{imgname}'
            img = cv2.imread(img_path)
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face)
                labels.append(userlist.index(user))  # Map the label to its index in userlist
                processed_images += 1
                print(f"Processing image {processed_images}/{total_images}")
            else:
                print(f"Failed to read image: {img_path}")
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    # Normalize images
    faces = faces.astype('float32') / 255.0
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        faces, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    # Define the CNN architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Add dropout layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Add dropout layer
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout layer
        Dense(len(userlist), activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Update progress at the end of each epoch
    def update_progress(epoch, logs):
        session['progress'] = (epoch + 1) / 10 * 100

    print_callback = LambdaCallback(on_epoch_end=update_progress)

    # Define the data augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[print_callback])
    
    # Save model
    model.save('static/cnn_recognition_model.h5')
    
    # Save the face embeddings for each user
    for user in userlist:
        user_embeddings = []
        for imgname in os.listdir(f'static/faces/CNNalgo/{user}'):
            if not imgname.endswith('.jpg') and not imgname.endswith('.jpeg') and not imgname.endswith('.png'):
                continue  # Skip .npy files
            
            img_path = f'static/faces/CNNalgo/{user}/{imgname}'
            img = cv2.imread(img_path)
            resized_face = cv2.resize(img, (50, 50))
            face_embedding = model.predict(resized_face.reshape(1, 50, 50, 3))[0]
            user_embeddings.append(face_embedding)
        user_embedding = np.mean(user_embeddings, axis=0)
        np.save(f'static/faces/CNNalgo/{user}/embedding.npy', user_embedding)

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
    userlist = os.listdir('static/faces/CNNalgo')
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
    session['progress'] = 0
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, algo='CNN', facereco = False)

identified_users = {}  # Format: {'person_name': (first_detection_time, last_detection_time, duration)}

# Adjust the 'start' function to track detection duration under one label
@app.route('/start', methods=['GET'])
def start():

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
            identified_person = identify_face(face.reshape(1, 50, 50, 3))
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
    return redirect('/')

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/CNNalgo/' + newusername + '_' + str(newuserid)
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
    userlist = os.listdir('static/faces/CNNalgo')  # Get list of users

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
    userlist = os.listdir('static/faces/CNNalgo')
    return jsonify(userlist)  # Return the list of users as a JSON response

# Add this route to remove a user
@app.route('/remove_user/<username>', methods=['POST'])
def remove_user(username):
    user_folder = os.path.join('static/faces/CNNalgo', username)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)  # Remove the directory
        return jsonify({'message': f'Successfully removed {username}'}), 200
    return jsonify({'message': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)