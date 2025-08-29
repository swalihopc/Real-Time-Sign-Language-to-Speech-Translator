from function import *
import cv2
import os

# Create directories for each action and store frames
for action in actions:  # actions should contain A-Z
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Initialize Mediapipe for hand detections
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as hands:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                frame = cv2.imread(f'Image/{action}/{sequence}.png')
            if frame is None:
               print('Warning: image is not found')
               continue

            # Perform hand detection
            image, results = mediapipe_detection(frame, hands)

            # Check for hands in the frame
            if results.multi_hand_landmarks:
               print(f'Hand Detected for {action} sequence {sequence} frame {frame_num}')
            else:
               print(f'Hand Not Detected for {action} sequence {sequence} frame {frame_num}')

            # Draw landmarks
            draw_styles_landmarks(image, results)

            # Display collection status
            message = f'Collecting frames for {action} Video {sequence}'
            cv2.putText(image, message, (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            # Extract and save keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            
            # Quit the loop when 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
               break
          
    # Close the OpenCV window
    cv2.destroyAllWindows()     