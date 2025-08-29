import threading
import time
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Config ---
MODEL_PATH = "sign_language_model.h5"        
CAM_INDEX = 0                   # change if you have multiple webcams
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_SIZE = (64, 64)             # model input size (change to your model input if different)
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

app = Flask(__name__)

# --- Globals for sharing prediction with web client ---
latest_label = {"letter": "", "confidence": 0.0, "timestamp": time.time()}

# --- Video capture in separate thread for performance ---
class VideoCamera:
    def __init__(self, cam_index=CAM_INDEX):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")
        self.lock = threading.Lock()
        self.running = True

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

        # Load model
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{MODEL_PATH}': {e}")

        # Start capture thread
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        self.frame = None

    def _update_loop(self):
        global latest_label
        while self.running:
            ret, raw_frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(raw_frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            predicted_letter = ""
            confidence = 0.0

            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # compute bounding box from landmarks
                x_coords = []
                y_coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    x_coords.append(int(lm.x * w))
                    y_coords.append(int(lm.y * h))
                x_min = max(min(x_coords) - 20, 0)
                x_max = min(max(x_coords) + 20, w)
                y_min = max(min(y_coords) - 20, 0)
                y_max = min(max(y_coords) + 20, h)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Extract ROI and preprocess
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size != 0:
                    try:
                        roi_resized = cv2.resize(roi, ROI_SIZE)
                        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                        roi_array = roi_rgb.astype("float32") / 255.0
                        roi_array = img_to_array(roi_array)
                        roi_array = np.expand_dims(roi_array, axis=0)  # (1,h,w,3)

                        preds = self.model.predict(roi_array)
                        idx = np.argmax(preds[0])
                        predicted_letter = LABELS[idx]
                        confidence = float(preds[0][idx])
                    except Exception as e:
                        # if prediction fails, keep blank
                        predicted_letter = ""
                        confidence = 0.0

            # Put text overlay
            text = f"{predicted_letter} ({confidence*100:.1f}%)" if predicted_letter else "..."
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # update latest_label in a thread-safe way
            with self.lock:
                self.frame = frame.copy()
                if predicted_letter:
                    latest_label = {"letter": predicted_letter,
                                    "confidence": confidence,
                                    "timestamp": time.time()}

            # small sleep to avoid hogging CPU
            time.sleep(0.01)

    def get_frame(self):
        # returns jpeg bytes of current frame
        with self.lock:
            if self.frame is None:
                # return a black frame placeholder
                black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', black)
                return jpeg.tobytes()
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        self.hands.close()

camera = VideoCamera()

# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest')
def latest():
    # return the latest predicted letter and confidence as JSON
    return jsonify(latest_label)

# graceful shutdown
import atexit
@atexit.register
def cleanup():
    try:
        camera.stop()
    except Exception:
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
