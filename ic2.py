import os
import cv2
import numpy as np
import pandas as pd
import mahotas
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract RGB color features
def extract_rgb_features(image):
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    return [r_mean, g_mean, b_mean]

# Function to extract Haralick texture features
def extract_haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick.tolist()

# Specify the main folder containing all the subfolders with labeled images
main_folder = '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set'

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

# Extract features
features = []
valid_labels = []
for img_path, label in zip(images, labels):
    print(f"Attempting to load image: {img_path}")
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Error: Could not load image at {img_path}")
        continue  # Skip to the next image if loading fails
    
    # Proceed with feature extraction only if the image is successfully loaded
    rgb_features = extract_rgb_features(image)
    haralick_features = extract_haralick_features(image)
    features.append(rgb_features + haralick_features)
    valid_labels.append(label)  # Only append labels for successfully loaded images

# Convert to DataFrame
columns = ['r_mean', 'g_mean', 'b_mean'] + [f'haralick_{i}' for i in range(13)]
df = pd.DataFrame(features, columns=columns)
df['label'] = valid_labels

# Encode labels
df['label'] = df['label'].astype('category').cat.codes

# Split data
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_preds = lda.predict(X_test)
lda_accuracy = accuracy_score(y_test, lda_preds)
print(f'LDA Accuracy: {lda_accuracy}')

# Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f'SVM Accuracy: {svm_accuracy}')

