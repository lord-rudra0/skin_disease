import ssl
import certifi

# Use certifi's certificate bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract deep features using ResNet50
def extract_deep_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    deep_features = base_model.predict(img_data)
    return deep_features.flatten()

# Specify the main folder containing all the subfolders with labeled images
main_folder = '/Users/vinayakprakash/Downloads/skin-disease-datasaet'  # Update with your dataset path

# Automatically gather all image paths and corresponding labels
images = []
labels = []
for subdir, _, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            img_path = os.path.join(subdir, file)
            label = os.path.basename(subdir)  # The label is the subfolder name
            images.append(img_path)
            labels.append(label)

# Extract features using ResNet50
features = []
valid_labels = []
for img_path, label in zip(images, labels):
    print(f"Processing image: {img_path}")
    try:
        deep_features = extract_deep_features(img_path)
        features.append(deep_features)
        valid_labels.append(label)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(features)
df['label'] = valid_labels

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split data
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Naive Bayes': GaussianNB()
}

# Perform hyperparameter tuning on SVM using grid search
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
svm = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
svm.fit(X_train, y_train)
print(f"Best SVM Parameters: {svm.best_params_}")

# Ensemble Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('lda', classifiers['LDA']),
    ('svm', svm.best_estimator_),
    ('nb', classifiers['Naive Bayes'])
], voting='hard')

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Ensemble Model Accuracy: {accuracy:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.show()


