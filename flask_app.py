'''import os
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded image
    img_path = os.path.join('static', secure_filename(file.filename))
    file.save(img_path)
    
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Perform inference using YOLOv5
    results = model(img)  # Inference
    
    # Render the results on the image
    annotated_img = results.render()[0]  # Get the annotated image as a NumPy array
    
    # Convert to PIL image for saving
    annotated_img_pil = Image.fromarray(annotated_img)
    
    # Save the annotated image
    annotated_img_path = os.path.join('static', 'annotated_' + secure_filename(file.filename))
    annotated_img_pil.save(annotated_img_path)
    return render_template('results.html', annotated_image_url=annotated_image_url)
    
    # Return the detection results and the path to the annotated image
    return jsonify({
        "results": results.pandas().xyxy[0].to_dict(orient="records"),
        "annotated_image_url": f"/static/annotated_{secure_filename(file.filename)}"
    })

if __name__ == '__main__':
    app.run(debug=True)
    
'''




'''
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded image
    img_path = os.path.join('static', secure_filename(file.filename))
    file.save(img_path)
    
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Perform inference using YOLOv5
    results = model(img)  # Inference
    
    # Render the results on the image
    annotated_img = results.render()[0]  # Get the annotated image as a NumPy array
    
    # Convert to PIL image for saving
    annotated_img_pil = Image.fromarray(annotated_img)
    
    # Save the annotated image
    annotated_img_path = os.path.join('static', 'annotated_' + secure_filename(file.filename))
    annotated_img_pil.save(annotated_img_path)
    
    # Define the correct URL for the annotated image
    annotated_image_url = f"/static/annotated_{secure_filename(file.filename)}"
    
    # Render the results.html template with the annotated image URL
    return render_template('results.html', annotated_image_url=annotated_image_url)
    
    # Return the detection results and the path to the annotated image
    return jsonify({
        "results": results.pandas().xyxy[0].to_dict(orient="records"),
        "annotated_image_url": annotated_image_url
    })

if __name__ == '__main__':
    app.run(debug=True)
'''
'''

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded image
    img_path = os.path.join('static', secure_filename(file.filename))
    file.save(img_path)
    
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Perform inference using YOLOv5
    results = model(img)  # Inference
    
    # Render the results on the image
    annotated_img = results.render()[0]  # Get the annotated image as a NumPy array
    
    # Convert to PIL image for saving
    annotated_img_pil = Image.fromarray(annotated_img)
    
    # Save the annotated image
    annotated_img_path = os.path.join('static', 'annotated_' + secure_filename(file.filename))
    annotated_img_pil.save(annotated_img_path)
    
    # Define the correct URL for the annotated image
    annotated_image_url = f"/static/annotated_{secure_filename(file.filename)}"
    
    # Extract detected entities (labels) and bounding box coordinates
    detected_entities = []
    
    for i, row in results.pandas().xyxy[0].iterrows():
        entity = row['name']  # Detected entity (class label)
        x = int(row['xmin'])  # Bounding box x-min
        y = int(row['ymin'])  # Bounding box y-min
        w = int(row['xmax']) - int(row['xmin'])  # Width
        h = int(row['ymax']) - int(row['ymin'])  # Height
        detected_entities.append([entity, x, y, w, h])

    # Return the detection results along with the path to the annotated image
    return jsonify({
        "results": detected_entities,
        "annotated_image_url": annotated_image_url
    })

if __name__ == '__main__':
    app.run(debug=True)


'''


'''
import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Load the BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')

# Create a templates folder to store HTML files
if not os.path.exists('templates'):
    os.makedirs('templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    img_path = os.path.join('static', filename)
    file.save(img_path)

    # Read the image using OpenCV
    img_cv2 = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Perform inference using YOLOv5
    yolo_results = yolo_model(img_rgb)  # Inference

    # Render the results on the image
    annotated_img = yolo_results.render()[0]  # Get the annotated image as a NumPy array

    # Convert to PIL image for saving
    annotated_img_pil = Image.fromarray(annotated_img)

    # Save the annotated image
    annotated_filename = 'annotated_' + filename
    annotated_img_path = os.path.join('static', annotated_filename)
    annotated_img_pil.save(annotated_img_path)

    # Define the correct URL for the annotated image
    annotated_image_url = f"/static/{annotated_filename}"

    # Extract detected entities (labels) and bounding box coordinates
    detected_entities = []

    for i, row in yolo_results.pandas().xyxy[0].iterrows():
        entity = row['name']  # Detected entity (class label)
        x = int(row['xmin'])  # Bounding box x-min
        y = int(row['ymin'])  # Bounding box y-min
        w = int(row['xmax']) - int(row['xmin'])  # Width
        h = int(row['ymax']) - int(row['ymin'])  # Height
        detected_entities.append([entity, x, y, w, h])

    # Generate a caption for the image using BLIP
    raw_image = Image.open(img_path).convert('RGB')
    blip_inputs = blip_processor(raw_image, return_tensors="pt")
    blip_out = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)

    # Render the results.html template with the annotated image URL, detected entities, and caption
    return render_template('results.html',
                           annotated_image_url=annotated_image_url,
                           detected_entities=detected_entities,
                           caption=caption)


if __name__ == '__main__':
    app.run(debug=True)
'''


'''
import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Load the BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')

# Create a templates folder to store HTML files
if not os.path.exists('templates'):
    os.makedirs('templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    img_path = os.path.join('static', filename)
    file.save(img_path)

    # Read the image using OpenCV
    img_cv2 = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Perform inference using YOLOv5
    yolo_results = yolo_model(img_rgb)  # Inference

    # Render the results on the image
    annotated_img = yolo_results.render()[0]  # Get the annotated image as a NumPy array

    # Convert to PIL image for saving
    annotated_img_pil = Image.fromarray(annotated_img)

    # Save the annotated image
    annotated_filename = 'annotated_' + filename
    annotated_img_path = os.path.join('static', annotated_filename)
    annotated_img_pil.save(annotated_img_path)

    # Define the correct URL for the annotated image
    annotated_image_url = f"/static/{annotated_filename}"

    # Extract detected entities (labels) and bounding box coordinates
    detected_entities = []
    for i, row in yolo_results.pandas().xyxy[0].iterrows():
        entity = row['name']  # Detected entity (class label)
        x = int(row['xmin'])  # Bounding box x-min
        y = int(row['ymin'])  # Bounding box y-min
        w = int(row['xmax']) - int(row['xmin'])  # Width
        h = int(row['ymax']) - int(row['ymin'])  # Height
        detected_entities.append([entity, x, y, w, h])

    # Generate a caption for the image using BLIP
    raw_image = Image.open(img_path).convert('RGB')
    blip_inputs = blip_processor(raw_image, return_tensors="pt")
    blip_out = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)

    # Render the results.html template with the annotated image URL, detected entities, and caption
    return render_template('results.html',
                           annotated_image_url=annotated_image_url,
                           detected_entities=detected_entities,
                           caption=caption)


if __name__ == '__main__':
    app.run(debug=True)


'''




import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for the upload route

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Load the BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Create a static folder to save images
if not os.path.exists('static'):
    os.makedirs('static')

# Create a templates folder to store HTML files
if not os.path.exists('templates'):
    os.makedirs('templates')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    img_path = os.path.join('static', filename)
    file.save(img_path)

    # Read the image using OpenCV
    img_cv2 = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Perform inference using YOLOv5
    yolo_results = yolo_model(img_rgb)  # Inference

    # Render the results on the image
    annotated_img = yolo_results.render()[0]  # Get the annotated image as a NumPy array
    annotated_img_pil = Image.fromarray(annotated_img)

    # Save the annotated image
    annotated_filename = 'annotated_' + filename
    annotated_img_path = os.path.join('static', annotated_filename)
    annotated_img_pil.save(annotated_img_path)

    # Define the correct URL for the annotated image
    annotated_image_url = f"/static/{annotated_filename}"

    # Extract detected entities (labels), confidence, and bounding box coordinates
    detected_entities = []
    for i, row in yolo_results.pandas().xyxy[0].iterrows():
        entity = row['name']  # Detected entity (class label)
        confidence = row['confidence']  # Confidence score
        x = int(row['xmin'])  # Bounding box x-min
        y = int(row['ymin'])  # Bounding box y-min
        w = int(row['xmax']) - int(row['xmin'])  # Width
        h = int(row['ymax']) - int(row['ymin'])  # Height
        detected_entities.append([entity, x, y, w, h])

    # Generate a caption for the image using BLIP
    raw_image = Image.open(img_path).convert('RGB')
    blip_inputs = blip_processor(raw_image, return_tensors="pt")
    blip_out = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)

    # Get the user's question and use BLIP to answer it if provided
    question = request.form.get("question", "")
    answer = ""
    if question:
        inputs = blip_processor(raw_image, question=question, return_tensors="pt")
        out = blip_model.generate(**inputs)
        answer = blip_processor.decode(out[0], skip_special_tokens=True)

    # Render the results.html template with the annotated image URL, detected entities, and caption
    return render_template('results.html',
                           annotated_image_url=annotated_image_url,
                           results=detected_entities,
                           generated_description=caption,
                           question=question,
                           answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
