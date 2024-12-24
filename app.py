import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import date, datetime, timedelta
import shutil
import face_recognition as fr
import pandas as pd
import threading
import numpy as np
import time
import pytz

from datetime import date, datetime, timedelta

app = Flask(__name__)

nimgs = 7  # Reduce the number of images for face registration to 7
unknown_threshold = 0.5  # Adjusted threshold for recognizing unknown faces

# Global variable to track progress
loading_progress = 0

last_detection = {}  

TIMEOUT_WINDOW = timedelta(minutes=2)  

def load_faces_and_update_progress():
    global loading_progress
    
    # Update known face encodings after adding a new user
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_face_encodings() 
    # Simulate progress for demonstration (replace with actual loading logic)
    for i in range(1, 101):
        loading_progress = i
        time.sleep(0.1)  # Simulate time taken for loading

# Create necessary directories and CSV file
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

attendance_csv = f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv'
if not os.path.isfile(attendance_csv):
    with open(attendance_csv, 'w') as f:
        f.write('Name,Roll,Time In,Time Out\n')  # Add 'Time In' and 'Time Out' headers

# Update the extract_attendance function to handle missing file
def extract_attendance():
    if not os.path.isfile(attendance_csv):
        return [], [], [], [], 0  # Return empty lists if file doesn't exist
    df = pd.read_csv(attendance_csv)
    
    # Replace NaN values with an empty string
    df.fillna('', inplace=True)
    
    names = df['Name']
    rolls = df['Roll']
    times_in = df['Time In']
    times_out = df['Time Out']
    l = len(df)
    return names, rolls, times_in, times_out, l

def add_attendance(name, action):
    if '_' not in name:
        return
    username, userid = name.split('_')

    # Set the timezone to Kuala Lumpur/Singapore
    kl_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    current_time = datetime.now(kl_timezone)
    current_time_str = current_time.strftime("%H:%M:%S")
    
    # Load the attendance CSV for today
    attendance_file = f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv'
    
    if not os.path.isfile(attendance_file):
        # If the CSV doesn't exist, create it with headers
        df = pd.DataFrame(columns=['Name', 'Roll', 'Time In', 'Time Out'])
    else:
        # Load the existing CSV file
        df = pd.read_csv(attendance_file)

    # Check if the user has already checked in for the day
    user_records = df[df['Roll'] == int(userid)]

    if action == 'in':
        # Log Time In if no previous record exists or last record has Time Out
        if user_records.empty or pd.notna(user_records.iloc[-1]['Time Out']):
            new_row = pd.DataFrame([[username, int(userid), current_time_str, '']], 
                                   columns=['Name', 'Roll', 'Time In', 'Time Out'])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Logged Time In for {username} at {current_time_str}")
    elif action == 'out':
        # Log Time Out only if there is a Time In that hasn't been followed by a Time Out
        if not user_records.empty and pd.isna(user_records.iloc[-1]['Time Out']):
            df.loc[df.index == user_records.index[-1], 'Time Out'] = current_time_str
            print(f"Logged Time Out for {username} at {current_time_str}")

    # Save the updated CSV back to the file
    df.to_csv(attendance_file, index=False)

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

def load_face_encodings():
    """Load all face encodings from the dataset"""
    known_encodings = []
    known_names = []
    userlist = os.listdir('face_rec')
    
    for user in userlist:
        for imgname in os.listdir(f'face_rec/{user}'):
            img_path = f'face_rec/{user}/{imgname}'
            image = fr.load_image_file(img_path)
            encoding = fr.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(user)
    return known_encodings, known_names

def identify_face(face_encoding):
    """Identify the face from known encodings."""
    if len(known_face_encodings) == 0:
        return "Unknown"  # If no encodings exist, return Unknown immediately
    
    # Compare faces and calculate distances
    matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=unknown_threshold)
    face_distances = fr.face_distance(known_face_encodings, face_encoding)

    # If no matches or distances exist, return Unknown
    if len(face_distances) == 0 or not any(matches):
        return "Unknown"

    # Find the best match
    best_match_index = np.argmin(face_distances)
    
    # Ensure that the best match has a sufficiently low distance
    if matches[best_match_index] and face_distances[best_match_index] < unknown_threshold:  # Use threshold dynamically
        return known_face_names[best_match_index]
    else:
        return "Unknown"

# Initialize a dictionary to store detection times
detection_start_time = {}

def process_frames(frame, face_data):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame to 1/4 size
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Face location detection
    face_locations = fr.face_locations(rgb_frame, model="hog")  
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]  # Scale back up
        name = identify_face(face_encoding)
        
        # Check if the face is not "Unknown" and record detection start time
        if name != "Unknown" and name not in detection_start_time:
            detection_start_time[name] = datetime.now()
        elif name != "Unknown":
            # Calculate detection duration
            detection_duration = datetime.now() - detection_start_time[name]
            if detection_duration >= MIN_DETECTION_DURATION:
                add_attendance(name)
                del detection_start_time[name]

        # Remove if detection exceeds timeout
        if name in detection_start_time and datetime.now() - detection_start_time[name] >= DETECTION_TIMEOUT:
            del detection_start_time[name]

        face_data.append((name, (left, top, right, bottom)))

@app.route('/')
def home():
    names, rolls, times_in, times_out, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, l=l, totalreg=len(os.listdir('face_rec')), datetoday2=date.today().strftime("%d-%B-%Y"), algo='FaceReco mod', facereco = True)

@app.route('/attendance_data', methods=['GET'])
def attendance_data():
    names, rolls, times_in, times_out, l = extract_attendance()
    return jsonify({
        'names': names.tolist(),
        'rolls': rolls.tolist(),
        'times_in': times_in.tolist(),
        'times_out': times_out.tolist(),
        'length': l
    })

# Define the minimum detection duration and detection timeout
MIN_DETECTION_DURATION = timedelta(seconds=3)  # 3 seconds
DETECTION_TIMEOUT = timedelta(seconds=2)  # 2 seconds

# Initialize a dictionary to store the detection start time for each face
detection_start_time = {}

# a detection that can be done from further away but a bit laggy
@app.route('/start', methods=['GET'])
def start():
    """Start the face recognition process"""
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width to 640 (or higher if your camera supports it)
    cap.set(4, 480)  # Set height to 480 (or higher)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS for smoother capture
    cv2.namedWindow('Attendance', cv2.WND_PROP_FULLSCREEN)
    global last_detection
    current_time = datetime.now()
    
    face_data = []
    process_interval = 5  # Process every 5 frames to reduce load
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame every 'process_interval' frames
        if frame_count % process_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use HOG model for faster face location detection
            face_locations = fr.face_locations(rgb_frame, model="hog")  
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            
            # Reset face_data only after processing new frame
            face_data = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = identify_face(face_encoding)
                face_data.append((name, (left, top, right, bottom)))

        # Display the results
        for name, (left, top, right, bottom) in face_data:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Only process known faces
        if name != "Unknown":
            if name not in last_detection:
                last_detection[name] = {'last_time_in': None, 'last_time_out': None}

            # Print the current state of last_detection
            print(f"Current state for {name} before processing: {last_detection[name]}")

            # Check if the user should be marked for Time In
            if (last_detection[name]['last_time_in'] is None or 
                (current_time - last_detection[name]['last_time_in']) >= TIMEOUT_WINDOW):
                
                # Allow a new Time In if the last Time Out was logged
                if last_detection[name]['last_time_out'] is not None:
                    last_detection[name]['last_time_in'] = None  # Reset last_time_in after Time Out

                if last_detection[name]['last_time_in'] is None and last_detection[name]['last_time_out'] is None:
                    add_attendance(name, 'in')  # Mark attendance for Time In
                    last_detection[name]['last_time_in'] = current_time       
                    print(f"Logged Time In for {name} at {current_time}")
                    
                if last_detection[name]['last_time_in'] is None and last_detection[name]['last_time_out'] is not None:
                    if (current_time - last_detection[name]['last_time_out']) >= TIMEOUT_WINDOW:
                        add_attendance(name, 'in')  # Mark attendance for Time In
                        last_detection[name]['last_time_in'] = current_time       
                        last_detection[name]['last_time_out'] = None
                        print(f"Logged Time In for {name} at {current_time}")

            # Check if the user should be marked for Time Out
            if last_detection[name]['last_time_out'] is None and last_detection[name]['last_time_in'] is not None:
                # Check if the timeout window has passed since last Time In
                if (current_time - last_detection[name]['last_time_in']) >= TIMEOUT_WINDOW:
                    add_attendance(name, 'out')  # Mark attendance for Time Out
                    last_detection[name]['last_time_out'] = current_time
                    last_detection[name]['last_time_in'] = None
                    print(f"Logged Time Out for {name} at {current_time}")
                else:
                    print(f"{name} has not yet timed out. Time since last Time In: {current_time - last_detection[name]['last_time_in']}")

        # Show the frame
        cv2.imshow('Attendance', frame)
        frame_count += 1
        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/add', methods=['POST'])
def add_user():
    """Add a new user and update face encodings"""
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'face_rec/{newusername}_{newuserid}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    cap = cv2.VideoCapture(0)
    i = 0
    
    while i < nimgs:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = fr.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, f'Capturing: {i+1}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(f"{userimagefolder}/image_{i+1}.jpg", face_image)
            i += 1
        
        cv2.imshow('Adding Faces', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/users/', methods=['GET'])
def display_users():
    userlist = os.listdir('face_rec')
    return jsonify(userlist)

@app.route('/remove_user/<username>', methods=['POST'])
def remove_user(username):
    user_folder = os.path.join('face_rec', username)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)
        return jsonify({'message': f'Successfully removed {username}'}), 200
    return jsonify({'message': 'User not found'}), 404

@app.route('/reload', methods=['GET'])
def reload():
    global loading_progress
    loading_progress = 0  # Reset progress

    # Start loading in a separate thread
    thread = threading.Thread(target=load_faces_and_update_progress)
    thread.start()

    return jsonify({'status': 'Datasets reloading started!'})

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify({'progress': loading_progress})

if __name__ == '__main__':
    app.run(debug=True)
