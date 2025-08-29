import cv2
import numpy as np
import os
import mediapipe as mp

# initialize the mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# perform mediapipe detection for images
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

# draw the landmarks and hand connections
def draw_styles_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

# extract the keypoints from detected landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        return rh

    return np.zeros(21 * 3)

# define paths and parameters for data detection
DATA_PATH = os.path.join('MP_Data')

# Full alphabet actions
actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

no_sequences = 30
sequence_length = 30

