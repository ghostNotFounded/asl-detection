def get_landmark_list(width, height, landmarks):
    landmark_points = []

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * width), width - 1)
        landmark_y = min(int(landmark.y * height), height - 1)

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points