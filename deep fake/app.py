from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile

app = Flask(__name__)

# Load the trained deepfake detection model
MODEL_PATH = 'D:\studyes\deep fake\inceptionv3_deepfake_model.h5'  # Update path
model = load_model(MODEL_PATH)

def preprocess_frame(frame):
    """Preprocess a frame for model prediction."""
    img = cv2.resize(frame, (256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_video(video_path):
    """Predict if a video is real or fake with frame visualization."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame
            img_array = preprocess_frame(frame)
            prediction = model.predict(img_array)[0][0]
            predictions.append(prediction)

            # Convert frame to Base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_data.append({"frame": frame_base64, "prediction": f"{prediction:.2f}"})

    cap.release()

    # Compute overall prediction
    if predictions:
        avg_prediction = np.mean(predictions)
        result = "Real" if avg_prediction > 0.5 else "Fake"
        probability = avg_prediction if result == "Real" else 1 - avg_prediction
        return {"result": result, "probability": f"{probability:.2f}", "frames": frames_data}
    else:
        return {"error": "No frames processed"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    video = request.files['video']
    
    if video.filename == '':
        return jsonify({'error': 'No selected file'})

    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
    video.save(temp_video_path)

    prediction_data = predict_video(temp_video_path)
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
