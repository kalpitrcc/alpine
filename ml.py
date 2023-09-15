import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from alibi_detect.utils.data import create_outlier_batch
from alibi_detect.od import OutlierAE
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Load your pretrained MNIST model
# Replace 'your_model_path' with the actual path to your model file.
model = tf.keras.models.load_model('your_model_path')

# Evaluate the model on the MNIST test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Accuracy on MNIST test set: {accuracy * 100:.2f}%')

# Generate adversarial examples
n_samples = 1000
X_outliers = create_outlier_batch(X_train, n_samples=n_samples, perc_outlier=10)

# Create an OutlierAE detector
detector = OutlierAE(threshold=0.2, model=model)

# Fit the detector to the training data
detector.fit(X_train)

# Detect outliers in the test set (adversarial examples)
preds = detector.predict(X_outliers)

# Calculate the detection performance
n_correct = np.sum(preds['data']['is_outlier'] == 1)
n_total = n_samples
detection_accuracy = n_correct / n_total
print(f'Detection accuracy on adversarial examples: {detection_accuracy * 100:.2f}%')

# Print classification report
print(classification_report(np.ones(n_samples), preds['data']['is_outlier'], target_names=['Normal', 'Anomaly']))

# Plot confusion matrix
conf_matrix = confusion_matrix(np.ones(n_samples), preds['data']['is_outlier'])
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['Normal', 'Anomaly']))
plt.xticks(tick_marks, ['Normal', 'Anomaly'], rotation=45)
plt.yticks(tick_marks, ['Normal', 'Anomaly'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
