import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = os.path.join(r'C:\Users\Artatrana\Downloads\action_detection\data_collection\MP_Data')
actions = np.array(['hello','thank you','yes', 'no', 'how are you', 'fine', 'goodbye'])
no_sequences = 30
sequence_length = 30

def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    return train_test_split(X, y, test_size=0.2)
