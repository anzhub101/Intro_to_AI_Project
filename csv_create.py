import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

multimodal_model = load_model('multimodal.h5')
binary_model = load_model('binary.h5')

# Load the data
data = pd.read_csv('data.csv')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)  # Normalize pixel values to [-1, 1]


multi_labels = data['Label'].values
multi_images = [preprocess_image(f'test_data/{fname}') for fname in data['Filename']]
multi_images = np.vstack(multi_images)
multi_preds = multimodal_model.predict(multi_images)
multi_preds_labels = np.argmax(multi_preds, axis=1)

# Encode labels
label_binarizer = LabelBinarizer()
multi_labels_encoded = label_binarizer.fit_transform(multi_labels)

# Performance Metrics for multi-class classifier
multi_accuracy = accuracy_score(multi_labels_encoded.argmax(axis=1), multi_preds_labels)
multi_conf_matrix = confusion_matrix(multi_labels_encoded.argmax(axis=1), multi_preds_labels)

print("Multi-Class Classifier Metrics:")
print("Accuracy:", multi_accuracy)
print("Confusion Matrix:\n", multi_conf_matrix)

# For binary classifier
binary_labels = (data['Label'] == 'TV').astype(int)
binary_images = [preprocess_image(f'test_data/{fname}') for fname in data['file']]
binary_images = np.vstack(binary_images)
binary_preds_prob = binary_model.predict(binary_images)
binary_preds = (binary_preds_prob > 0.5).astype(int)

binary_accuracy = accuracy_score(binary_labels, binary_preds)
binary_precision = precision_score(binary_labels, binary_preds)
binary_recall = recall_score(binary_labels, binary_preds)
binary_tn, binary_fp, binary_fn, binary_tp = confusion_matrix(binary_labels, binary_preds).ravel()
binary_specificity = binary_tn / (binary_tn + binary_fp)
binary_fpr = binary_fp / (binary_fp + binary_tn)

print("\nBinary Classifier Metrics:")
print("Accuracy:", binary_accuracy)
print("Precision:", binary_precision)
print("Recall:", binary_recall)
print("Specificity:", binary_specificity)
print("False Positive Rate:", binary_fpr)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(binary_labels, binary_preds_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(binary_labels, binary_preds_prob)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()
