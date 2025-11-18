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

# Dynamically discover image files under the local dataset folder so this script
# is portable (avoid hard-coded absolute paths).
main_folder = 'skin-disease-datasaet'
images = []
labels = []
for subdir, _, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # strip filename whitespace to avoid trailing-space issues that
            # appeared in some committed lists (e.g. '... (54).jpg ').
            filename = file.strip()
            img_path = os.path.join(subdir, filename)
            images.append(img_path)
            # label by the containing folder name and strip extra whitespace
            labels.append(os.path.basename(subdir).strip())

# Diagnostics
print(f"Discovered {len(images)} image files under '{main_folder}'.")
if len(images) > 0:
    print("Sample paths:")
    for p in images[:5]:
        print(" -", p)
else:
    raise FileNotFoundError(f"No image files found under '{main_folder}'. Check that the dataset folder exists and contains images.")

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