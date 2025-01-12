import cv2
import numpy as np
import os
import time
from utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    create_action_folders,
    save_keypoints
)
import mediapipe as mp


DATA_PATH = os.path.join(r'C:\Users\Artatrana\Downloads\action_detection\data_collection\MP_Data')
actions = np.array(['hello','thank you','yes', 'no', 'how are you', 'fine', 'goodbye'])
no_sequences = 30
sequence_length = 30

create_action_folders(DATA_PATH, actions, no_sequences)

mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                
                draw_styled_landmarks(image, results)
                
                keypoints = extract_keypoints(results)
                
                save_keypoints(DATA_PATH, action, sequence, frame_num, keypoints)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', 
                            (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)
                #time.sleep(0.1)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
