import cv2
import numpy as np
import time
from keras.models import load_model
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz

model = load_model('action.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['hello', 'thank you', 'yes', 'no'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

sequence = []
sentence = []
timeout = 2  # seconds
threshold = 0.8
last_word_time = time.time()

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        # Collect keypoints for the sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        # Make predictions when sequence is complete
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = actions[np.argmax(res)]
            
            # If prediction confidence is high enough
            if res[np.argmax(res)] > threshold:
                # Check if the new word differs from the last one or if timeout has passed
                if len(sentence) == 0 or (action != sentence[-1] and time.time() - last_word_time > timeout):
                    sentence.append(action)
                    last_word_time = time.time()
            
            # Display only the last 5 words in the sentence
            display_sentence = ' '.join(sentence[-5:])
            image = prob_viz(res, actions, image, colors)
        else:
            # Initialize display_sentence if no sequence is ready
            display_sentence = ' '.join(sentence[-5:])

        # Display the constructed sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, display_sentence, (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
