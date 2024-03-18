#final project app for realtime face detection and antispoofing frontend into index.html
import base64
import datetime
import logging
import os
import time
from pytz import timezone
import cv2
import face_recognition
import numpy as np
import requests
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS

import dlib

import asyncio
import time
# Import the 'test' function from your existing code

indian_timezone = timezone('Asia/Kolkata')

# Configure logging
logging.basicConfig(level=logging.INFO, filename="file.log", filemode="a", format="%(asctime)s%(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z" ) # Customize the date format he)
logger = logging.getLogger()


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)
# sio = socketio.Server()
user_data = {}
known_face_encodings = []
known_face_names=[]


@app.route('/')
def index():
    return "<h3>welcome to the face api</h3>"


def take_encodings_image(user_id, environment="dev"):

    global known_face_encodings

    try:
        # Determine the URL based on the environment flag
        if environment == 'prod':
            api_url = f"http://13.126.129.218:7002/api/auth/user-profile-image/{user_id}"
        else:
            api_url = f"http://13.126.129.218:6002/api/auth/user-profile-image/{user_id}"
        # Hit the API with the userid
        # api_url = f"http://13.126.129.218:6002/api/auth/user-profile-image/{user_id}"
        response = requests.get(api_url)
        print("===response", response)
        logging.info('response: %s', response)
        # Check the API response and perform actions accordingly
        if response.status_code == 200:
            api_data = response.json()
            # print("===>apidata", api_data)
            profile_image_data = api_data.get('data', [])[0].get('profileImage', None)
            # print(profile_image_data)
            user_id = api_data.get('data', [])[0].get('userId', None)
            user_name = api_data.get('data', [])[0].get('userName', None)
            known_face_names.append(user_name)
            binary_data = base64.b64decode(profile_image_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            # Check if a face is detected in the image
            encoding_kn = face_recognition.face_encodings(image_np)[0]
            logging.info('face encodings length of known encoding : %s', len(encoding_kn))

            known_face_encodings.append(encoding_kn)

            #addition for new approach as deepface can check with two images
            known_path = f'images/known_{user_id}.jpg'
            os.makedirs(os.path.dirname(known_path), exist_ok=True)
            cv2.imwrite(known_path, image_np)
            # known_face_encodings[user_id] = {"user_name" :  user_name, "encoding": encoding_kn}
            print("====known_face_encoding", known_face_encodings)
            # if user_id in image_database:
            #     image_database[user_id][sid] = face_encoding
            # else:
            #     image_database[user_id] = face_encoding
            # print("===imagedatabse", image_database)
            return known_face_encodings

    except Exception as e:
        logging.error(f"Error: {e}")



# Load known faces and their encodings (replace with your data)
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        # Load the image from the request
        file = request.files['image']
        print("===>file", file)
        logging.info('file of unknown image : %s', file)

        user_id = request.form.get('user_id')
        print("-----userid", user_id)

        logging.info('user_id : %s', user_id)
        take_encodings_image(user_id)

        unknown_image = face_recognition.load_image_file(file)
        # print("====>unknown image", unknown_image)
        # # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image)
        print("---->face location", face_locations)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        print("------>faceencoding", len(face_encodings))
        logging.info('face encodings length of unknown encoding: %s', len(face_encodings))

        if len(face_encodings) > 0:
            # Loop through each face found in the unknown image
            results = []
            #
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                print("---->matches", matches)
                logging.info('matched result: %s', matches)

                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print("====>facedistance", face_distances)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(name)
                    logging.info('name of matched face : %s', name)

                    results.append({'name': name, "user_id": user_id,'is_known': True, 'status_code': 200, 'status': 'success', 'message': 'Match Found!'})

                else:
                    results.append({'name': name, 'is_known': False, 'status_code': 400, 'status': 'fail',
                                    'message': 'Match not found.'})

            logging.info('final response of results : %s', results)
            return jsonify({'results': results})
        else:
            return jsonify({'error': "Does not get face Encoding", 'status': 400})

    except Exception as e:
        return jsonify({'error': str(e)})





from deepface import DeepFace
# Specify the desired width and height
width = 640  # Set your desired width
height = 480  # Set your desired height
# working=====>
@app.route('/deep_face_check', methods=["POST"])
def deep_fc():
    file = request.files['image']
    if not file:
        return jsonify({"status": False, "message": 'Please provide file image', "statusCode":400})
    print("===>file", file)
    logging.info('file of unknown image : %s', file)
    user_id = request.form.get('user_id')
    print("-----userid", user_id)
    environment = request.form.get("environment")
    print(environment)
    if not user_id:
        return jsonify({"status": False, "message": 'Please provide user_id', "statusCode":400 })
    logging.info('user_id : %s', user_id)

    take_encodings_image(user_id, environment=environment)
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        live_frame_path = f'images/file1_{user_id}.jpg'
        os.makedirs(os.path.dirname(live_frame_path), exist_ok=True)
        # Write the image to the local directory
        cv2.imwrite(live_frame_path, image)

        # Proceed with face recognition if not spoofed
        known_image_path = f'images/known_{user_id}.jpg'
        # Pass the image path to DeepFace.verify
        custom_threshold = 0.4
        result_recognition = DeepFace.verify(img1_path=known_image_path , img2_path=live_frame_path, model_name='Facenet', distance_metric='cosine', enforce_detection=False )
        print("result_recognition", result_recognition)
        logging.info('result_recognition in deepface api : %s', result_recognition)
    # Your logic to handle the result
        if result_recognition['distance'] < custom_threshold:
            result_recognition['verified'] = True
            result = {'matched': True, 'message': 'Match Found!!', "status": True,  'statusCode': 200, 'user_id': user_id, "user": user_id}
            print(result)
            logging.info('result of deepface api true face: %s', result)
            return result
        else:
            result_recognition['verified'] = False
            result = {'matched': False, 'message': 'not match with db image!!', "status": False,  'statusCode': 400, 'user_id': "Unknown!", "user": "Unknown!"}
            logging.info('result of deepface api false face: %s', result)
            return result

    # Emit the result to the client
    #     sio.emit('face_recognition_result', result, room=sid)

    # Remove the temporary image file

    except  Exception as e:
        return jsonify({"statusCode": 500, "message": f"An error occurred: {str(e)}"})



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)