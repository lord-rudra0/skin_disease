import os
import cv2
import numpy as np
import pandas as pd
import mahotas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

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

# Splitting the dataset into features (X) and target (y)
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classifiers
classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SVC(kernel='linear', random_state=42),
    'ANN': MLPClassifier(hidden_layer_sizes=(100,), alpha=0.1, max_iter=200, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Accuracy storage
accuracy_results = []

# Evaluating each classifier
for name, clf in classifiers.items():
    # Training the model
    clf.fit(X_train, y_train)
    
    # Predicting on test data
    y_pred = clf.predict(X_test)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results.append((name, accuracy * 100))
    
    # Display classification report and confusion matrix
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Displaying the accuracies in tabular form
accuracy_df = pd.DataFrame(accuracy_results, columns=['Classifier', 'Accuracy (%)'])
print("\nAccuracies of Classifiers:")
print(accuracy_df)

# Plotting the accuracies for comparison
sns.barplot(x='Classifier', y='Accuracy (%)', data=accuracy_df)
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 100)
plt.show()

# Comparing with research paper values
research_values = {
    'Classifier': ['LDA', 'SVM', 'ANN', 'Naive Bayes'],
    'Accuracy (%)': [82.58, 80.65, 75.16, 72.58]  # Replace with values from the research paper
}

research_df = pd.DataFrame(research_values)

# Merging and displaying both results
comparison_df = accuracy_df.merge(research_df, on='Classifier', suffixes=('_Current', '_Research'))
print("\nComparison with Research Paper Values:")
print(comparison_df)

# Plotting the comparison
comparison_df.set_index('Classifier').plot(kind='bar')
plt.title('Classifier Accuracy Comparison with Research Paper')
plt.ylabel('Accuracy (%)')
plt.show()
