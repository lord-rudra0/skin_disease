import os
import ssl
import certifi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from scipy.stats import uniform

# Fix SSL certificate issue
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Load pre-trained EfficientNetB0 model + higher level layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# Function to extract deep features using EfficientNetB0
def extract_deep_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    deep_features = base_model.predict(img_data)
    return deep_features.flatten()

# Specify the main folder containing all the subfolders with labeled images
main_folder = '/Users/vinayakprakash/Downloads/skin-disease-datasaet/train_set'  # Update with your dataset path

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

# Extract features using EfficientNetB0
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

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SVC(kernel='linear', C=37.55, gamma='scale', class_weight='balanced', probability=True),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Perform hyperparameter tuning on Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1)
rf.fit(X_train, y_train)
print(f"Best Random Forest Parameters: {rf.best_params_}")

# Ensemble Voting Classifier with 'soft' voting for ROC curve
ensemble_model = VotingClassifier(estimators=[
    ('lda', classifiers['LDA']),
    ('svm', classifiers['SVM']),
    ('nb', classifiers['Naive Bayes']),
    ('rf', rf.best_estimator_)
], voting='soft')  # Use soft voting to get probabilities

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate performance
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Ensemble Model Accuracy: {accuracy:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities for ROC curve
y_prob = ensemble_model.predict_proba(X_test)

# Binarize the labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {le.classes_[i]} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Cross-validation to ensure robustness
scores = cross_val_score(ensemble_model, X, y, cv=10, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(scores):.2f}%")