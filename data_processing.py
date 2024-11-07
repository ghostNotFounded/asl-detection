import os
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm
import csv

from utils import get_landmark_list
from utils import pre_process_landmark

TRAIN_PATH = "./data/train"
signs = os.listdir(TRAIN_PATH)

IMAGE_FILES = []

dict = {}

with open("./data/landmarks_label.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for idx, sign in tqdm(enumerate(signs[1:]), total=len(signs[1:]), desc="Loading images"):
      PATH = f"{TRAIN_PATH}/{sign}"
      images = os.listdir(PATH)

      writer.writerow([sign])
      dict[sign] = idx

      for image in images:
          IMAGE_FILES.append(f"{PATH}/{image}")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


with open("./data/landmarks.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')  

    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
        
        for idx, file in tqdm(enumerate(IMAGE_FILES), total=len(IMAGE_FILES), desc="Processing Images"):
            sign = file.split("/")[3]

            image = cv.flip(cv.imread(file), 1)
            results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape

            for hand_landmarks in results.multi_hand_landmarks:
                landmark_points = get_landmark_list(width=image_width, height=image_height, landmarks=hand_landmarks.landmark)              
                processed_landmark_points = pre_process_landmark(landmark_points)

                writer.writerow([dict[sign]] + processed_landmark_points)



