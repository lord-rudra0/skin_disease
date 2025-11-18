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

# List of images with correct file paths
images = [
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (1).webp',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (2).png',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (3).jpeg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (17).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (18).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (20).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (21).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (27).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (36).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (40).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (50).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (55).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (64).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (66).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (69).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (73).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (83).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (84).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (98).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (106).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (119).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (147).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (155).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/2_BA-impetigo (16).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/9_BA-impetigo (42).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/10_BA-impetigo (89).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/18_BA-impetigo (82).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/22_BA-impetigo (55).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/27_BA-impetigo (54).jpg ',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/28_BA-impetigo (6).jpeg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/29_BA-impetigo (22).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/31_BA-impetigo (26).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/41_BA-impetigo (89).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/42_BA-impetigo (2).png',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/49_BA-impetigo (17).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/50_BA-impetigo (38).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/59_BA-impetigo (6).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/60_BA-impetigo (3).png',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/62_BA-impetigo (23).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/64_BA-impetigo (2).png',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/88_BA-impetigo (67).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/89_BA-impetigo (86).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/BA-impetigo/98_BA-impetigo (74).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (1).jpeg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (1).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (1).png',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (2).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (3).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (4).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (5).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (6).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (7).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (8).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (9).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (10).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (11).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (12).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (13).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (14).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (15).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (16).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (17).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (18).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (19).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (20).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (21).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (22).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (23).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (24).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (25).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (26).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (27).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (28).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (29).jpg',
    '/Users/vinayakprakash/Downloads/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (30).jpg'
]

# Corresponding labels
labels = [
    'BA-cellulitis(1)', 'BA-cellulitis(2)', 'BA-cellulitis(3)',
    'cellulitis (17)', 'cellulitis (18)', 'cellulitis (20)',
    'cellulitis (21)', 'cellulitis (27)', 'cellulitis (36)',
    'cellulitis (40)', 'cellulitis (50)', 'cellulitis (55)',
    'cellulitis (64)', 'cellulitis (66)', 'cellulitis (69)',
    'cellulitis (73)', 'cellulitis (83)', 'cellulitis (84)',
    'cellulitis (98)', 'cellulitis (106)', 'cellulitis (119)',
    'cellulitis (147)', 'cellulitis (155)','2_BA-impetigo (16)',
]

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