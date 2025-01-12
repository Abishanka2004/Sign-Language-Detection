

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# In[122]:


result_test = extract_keypoints(results)


# In[123]:


result_test


# In[115]:


468*3+33*4+21*3+21*3


# In[124]:


np.save('0', result_test)


# In[125]:


np.load('0.npy')


# # 4. Setup Folders for Collection

# In[9]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


# In[ ]:


# hello
## 0
## 1
## 2
## ...
## 29
# thanks

# I love you


# In[159]:


for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# # 5. Collect Keypoint Values for Training and Testing

# In[160]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


# In[158]:


cap.release()
cv2.destroyAllWindows()


# # 6. Preprocess Data and Create Labels and Features

# In[6]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[10]:


label_map = {label:num for num, label in enumerate(actions)}


# In[163]:


label_map


# In[65]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[66]:


np.array(sequences).shape


# In[67]:


np.array(labels).shape


# In[68]:


X = np.array(sequences)


# In[69]:


X.shape


# In[70]:


y = to_categorical(labels).astype(int)


# In[ ]:


y


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[73]:


y_test.shape


# # 7. Build and Train LSTM Neural Network

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[12]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[ ]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[ ]:


res = [.7, 0.2, 0.1]


# In[ ]:


actions[np.argmax(res)]


# In[14]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])


# In[79]:


model.summary()


# # 8. Make Predictions

# In[196]:


res = model.predict(X_test)


# In[213]:


actions[np.argmax(res[4])]


# In[214]:


actions[np.argmax(y_test[4])]


# # 9. Save Weights

# In[216]:


model.save('action.h5')


# In[217]:


del model


# In[15]:


model.load_weights('action.h5')


# # 10. Evaluation using Confusion Matrix and Accuracy

# In[16]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[90]:


yhat = model.predict(X_test)


# In[91]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[92]:


multilabel_confusion_matrix(ytrue, yhat)


# In[93]:


accuracy_score(ytrue, yhat)


# # 11. Test in Real Time

# In[21]:


colors = [(245,117,16), (117,245,16), (16,117,245)]
