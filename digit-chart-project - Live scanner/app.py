import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = tf.keras.models.load_model('my_digit_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

# --- TAB 1: SINGLE DIGIT ---
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            roi = img[y:y+h, x:x+w]
            scale = 20.0 / max(w, h)
            roi_res = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            pad_t, pad_b = (28 - roi_res.shape[0]) // 2, 28 - roi_res.shape[0] - ((28 - roi_res.shape[0]) // 2)
            pad_l, pad_r = (28 - roi_res.shape[1]) // 2, 28 - roi_res.shape[1] - ((28 - roi_res.shape[1]) // 2)
            img = cv2.copyMakeBorder(roi_res, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(img, (28, 28))
        
        img = img.astype('float32') / 255.0
        prediction = model.predict(img.reshape(1, 28, 28, 1))
        return jsonify({'prediction': int(np.argmax(prediction[0])), 'probabilities': [float(p) for p in prediction[0]]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- TAB 2: MULTI-DIGIT CANVAS ---
@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    try:
        data = request.get_json()
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = sorted([cv2.boundingRect(c) for c in contours], key=lambda r: r[0])
        
        final_pred = ""
        for x, y, w, h in rects:
            if w > 10 and h > 10:
                roi = img[y:y+h, x:x+w]
                scale = 20.0 / max(w, h)
                roi_res = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                pad_t, pad_b = (28 - roi_res.shape[0]) // 2, 28 - roi_res.shape[0] - ((28 - roi_res.shape[0]) // 2)
                pad_l, pad_r = (28 - roi_res.shape[1]) // 2, 28 - roi_res.shape[1] - ((28 - roi_res.shape[1]) // 2)
                digit_img = cv2.copyMakeBorder(roi_res, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                pred = model.predict((digit_img.astype('float32') / 255.0).reshape(1, 28, 28, 1))
                final_pred += str(int(np.argmax(pred[0])))
        return jsonify({'prediction': final_pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- TAB 3: WEBCAM BOUNDING BOXES & HIGH ACCURACY ---
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # 1. Blur to remove noise
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        
        # 2. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. Morphological Closing (Connects broken pen strokes!)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = sorted([cv2.boundingRect(c) for c in contours], key=lambda r: r[0])
        
        results = []
        
        for x, y, w, h in rects:
            # Ignore very small or very large boxes
            if 15 < w < 200 and 20 < h < 200:
                roi = thresh[y:y+h, x:x+w]
                scale = 20.0 / max(w, h)
                roi_res = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                pad_t, pad_b = (28 - roi_res.shape[0]) // 2, 28 - roi_res.shape[0] - ((28 - roi_res.shape[0]) // 2)
                pad_l, pad_r = (28 - roi_res.shape[1]) // 2, 28 - roi_res.shape[1] - ((28 - roi_res.shape[1]) // 2)
                digit_img = cv2.copyMakeBorder(roi_res, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                
                pred = model.predict((digit_img.astype('float32') / 255.0).reshape(1, 28, 28, 1))
                digit = int(np.argmax(pred[0]))
                
                # Send back the coordinates AND the predicted number
                results.append({'x': x, 'y': y, 'w': w, 'h': h, 'digit': digit})
        
        return jsonify({'boxes': results})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)