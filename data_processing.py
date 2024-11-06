import os
import copy
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm
import csv
import itertools
from time import sleep

TRAIN_PATH = "./data/train"
signs = os.listdir(TRAIN_PATH)

IMAGE_FILES = []

for sign in tqdm(signs[1:], total=len(signs[1:]), desc="Loading data"):
    PATH = f"{TRAIN_PATH}/{sign}"
    images = os.listdir(PATH)

    for image in images[:100]:
        IMAGE_FILES.append(f"{PATH}/{image}")

sleep(1)
os.system("cls")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_landmark_list(width, height, landmarks):
    landmark_points = []

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * width), width - 1)
        landmark_y = min(int(landmark.y * height), height - 1)

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


with open("data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')  

    header = []
    for idx in range(21):
      header.append(f"x{idx + 1}")
      header.append(f"y{idx + 1}")
              
    writer.writerow(["Label"] + header)  

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

                writer.writerow([sign] + processed_landmark_points)



