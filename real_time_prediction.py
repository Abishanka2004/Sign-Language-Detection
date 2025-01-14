import cv2
import numpy as np
import os
from keras.models import load_model
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz

model = load_model('action.h5')


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


actions = np.array(['hello','thank you','yes', 'no', 'how are you', 'fine', 'goodbye']) 
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        
        draw_styled_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        
        sequence.append(keypoints)
        sequence = sequence[-30:]  
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                    
            if len(sentence) > 5:
                sentence = sentence[-5:]
            
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
