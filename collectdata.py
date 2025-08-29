import os
import cv2

# Save directory
directory = 'Image/'

# Create folders for A–Z if they don't exist
for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Press A–Z to capture gesture images. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Flip frame to fix mirror
    frame = cv2.flip(frame, 1)

    # Count images in each folder
    count = {letter.lower(): len(os.listdir(os.path.join(directory, letter)))
             for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]}

    # Draw ROI rectangle
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.putText(frame, "Press A–Z to capture, ESC to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show camera feed
    cv2.imshow("Data Collection", frame)
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)

    # Keyboard input
    key = cv2.waitKey(10) & 0xFF

    # Save image if key is a letter
    if ord('a') <= key <= ord('z'):
        letter = chr(key).upper()
        img_path = os.path.join(directory, letter, f"{count[letter.lower()] + 1}.png")
        cv2.imwrite(img_path, roi)
        print(f"📸 Saved {img_path}")

    # Exit if ESC pressed
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
