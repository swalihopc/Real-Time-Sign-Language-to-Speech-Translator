# 🖐️ **Real-Time Sign Language to Speech Translator**

This project captures real-time sign language gestures (A–Z) using a webcam, classifies them with a trained Deep Learning model (CNN + LSTM), and converts the output into speech. The system uses Mediapipe for hand detection and runs via a Flask web interface with live video streaming.

## ⚙️ Installation

Clone the repository
```
git clone https://github.com/swalihopc/Real-Time-Sign-Language-to-Speech-Translator.git
cd Real-Time-Sign-Language-to-Speech-Translator
```

Create a virtual environment
```
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

Install dependencies
```
pip install -r requirements.txt
```

## 📊 Dataset

- Dataset must be organized into subfolders for each letter (A–Z):

- Use collectdata.py to capture new gesture images via webcam.

- Each key press (A–Z) saves an image into the respective folder.

## 🧠 Training

- Load dataset & preprocess with data.py and function.py

- Extract hand landmarks using Mediapipe

- Train a CNN + LSTM model

- Save trained model as sign_language_model.h5

## 🚀 Running the Translator

Start the Flask app:
```
python app.py
```

Open your browser at:
👉 http://127.0.0.1:5000/

You will see:
✅ Live webcam feed
✅ Bounding box on detected hand
✅ Predicted letter overlay with confidence score
✅ Speech output of recognized letters

## 📌 Future Enhancements

- Recognize words & sentences, not just letters

- Improve accuracy with larger datasets

- Deploy as a mobile/desktop app

- Add multi-language text-to-speech
