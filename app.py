import mediapipe as mp

from utils import get_landmark_list, pre_process_landmark

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

labels_path = './data/landmarks_label.csv'
model_save_path = './models/classifier.keras'

model = load_model(f"./{model_save_path}")
input_shape = (21 * 2, )

labels = np.loadtxt(labels_path, delimiter=',', dtype=str, usecols=(0))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
prev_time = 0

# Initialize the sentence and previous label
sentence = ""
prev_label_text = ""

# ASL letter map with additional commands for "space" and "delete"
map = dict(enumerate(labels.flatten()))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
    
    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Time and FPS calculations
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        image = cv2.flip(image, 1)

        # Convert the image to RGB and process it with MediaPipe Hands
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand landmarks and calculate predictions
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label_text = "Nothing"

        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_points = get_landmark_list(width=image_width, height=image_height, landmarks=hand_landmarks.landmark)              
                processed_landmark_points = pre_process_landmark(landmark_points)

                # Prepare landmarks for prediction
                landmarks = np.array(processed_landmark_points).reshape(1, -1)
                letter_probs = model.predict(landmarks, verbose=False)
                letter_index = np.argmax(letter_probs)

                # Update label if confidence is high
                if letter_probs[0][letter_index] > 0.95:
                    temp_label_text = map[letter_index]
                    
                    # Handle "delete" action
                    if temp_label_text == "del":
                        sentence = sentence[:-1]
                    
                    # Handle "space" action
                    elif temp_label_text == "space":
                        sentence += " "
                    
                    # Add new letter if different from previous one
                    elif temp_label_text != prev_label_text:
                        sentence += temp_label_text
                        prev_label_text = temp_label_text

                    label_text = temp_label_text
                
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display FPS and predicted letter
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(image, fps_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label_text, (10, 60), font, 1, (255, 0, 255), 2, cv2.LINE_AA)

        # Display the sentence as it's constructed
        if sentence:
            print("Sentence:", sentence)

        cv2.imshow("ASL Translator", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
