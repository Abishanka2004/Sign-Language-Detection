
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from preprocess_data import load_data
from sklearn.metrics import confusion_matrix, classification_report

_, X_test, _, y_test = load_data()

model = load_model('action.h5')

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

actions = np.array(['hello', 'thank you', 'yes', 'no', 'how are you', 'fine', 'goodbye'])

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=actions))
