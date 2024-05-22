"""
NOTE: TO GET A MORE ACCURATE DETECTION TRAIN THE yolov8.pt MODEL FURTHER
Basic object detection for Videos and Images using yolov8.pt - pretrained model 
Author: Ishaan Ratanshi
Club Project for Google Developer Student Club
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os, cv2

OPENAI_API_KEY = "INSERT API KEY"
client = OpenAI(api_key=OPENAI_API_KEY)
model_id = 	'gpt-3.5-turbo-instruct'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
detected_objects = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/ishaanratanshi/Object Detection App/upload'
CORS(app)

def process_image_with_yolo(image_path):
    model = YOLO("yolov8n.pt")  # Initialize YOLO model
    img = cv2.imread(image_path)
    results = model.predict(img)
    names = model.names
    global detected_objects
    detected_objects = []

    for r in results:
        for c in r.boxes.cls:
            detected_objects.append(names[int(c)])
    if len(detected_objects) == 0:
        detected_objects.append("No objects detected")
    return detected_objects

def process_video_with_yolo(video_path):
    model = YOLO("yolov8n.pt")
    names = model.names
    global detected_objects
    detected_objects = []
    try:
        video = cv2.VideoCapture(video_path)

        while video.isOpened():
            ret, frame = video.read()
            results = model.predict(frame)
            print(results)
            for r in results:
                for c in r.boxes.cls:
                    detected_objects.append(names[int(c)])
                    if len(detected_objects) == 0:
                        detected_objects.append("No objects detected")
                    return detected_objects
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['POST', 'GET'])
def upload():
    global detected_objects  # Use the global variable

    if request.method == 'GET':
        return render_template("upload.html")

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    file_extension = file.filename.split('.')[-1].lower()

    if file_extension in ALLOWED_IMAGE_EXTENSIONS:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        detected_objects = process_image_with_yolo(image_path)
        return jsonify({'Detected objects': detected_objects})

    elif file_extension in ALLOWED_VIDEO_EXTENSIONS:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        detected_objects = process_video_with_yolo(video_path)  

        try:
            gpt_input = "Detected objects: " + ", ".join(detected_objects)
            response = client.completions.create(
                model=model_id,
                prompt=gpt_input,
                max_tokens=10
            )
            bot_reply = response.choices[0].text.strip()

            return jsonify({'Detected objects': detected_objects, 'reply': bot_reply})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Unsupported file format'}), 400

@app.route('/chat', methods=['POST'])
def process_message():
    global detected_objects  # Access the global variable
    try:
        gpt_input = "Detected objects: " + ", ".join(detected_objects)
        response = client.completions.create(
            model=model_id,
            prompt=gpt_input,
            max_tokens=100
        )
        bot_reply = response.choices[0].text.strip()

        response_data = {'Detected objects': detected_objects, 'reply': bot_reply}
        print("Response to client:", response_data)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
