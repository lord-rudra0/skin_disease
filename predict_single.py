#!/usr/bin/env python3
"""Minimal single-image predictor that prints predicted class and a binary
"skin disease: yes/no" decision based on confidence threshold.

Usage examples:
  python3 predict_single.py --image path/to/img.jpg
  python3 predict_single.py --image path/to/img.jpg --threshold 0.6

This script expects the following artifacts in the current directory:
  - ensemble_model.joblib
  - scaler.joblib
  - label_encoder.joblib (optional)

The script will load EfficientNetB0 to extract deep features, scale them,
and run the saved ensemble classifier. Output includes predicted class,
confidence and a short "Skin disease: YES/NO" line.
"""

import argparse
import os
import sys
import joblib
import numpy as np

from PIL import Image

try:
    # TensorFlow imports can be somewhat verbose; import only what we need
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as e:
    print("Error importing TensorFlow/Keras. Make sure your environment has TensorFlow installed.")
    raise


def load_artifacts(root_dir="."):
    model_path = os.path.join(root_dir, "ensemble_model.joblib")
    scaler_path = os.path.join(root_dir, "scaler.joblib")
    le_path = os.path.join(root_dir, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Required artifacts not found. Run the training script to create ensemble_model.joblib and scaler.joblib in project root.")

    ensemble = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    label_encoder = None
    classes = None
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
        try:
            classes = list(label_encoder.classes_)
        except Exception:
            classes = None

    # If label encoder not available, try to infer from the model
    if classes is None:
        if hasattr(ensemble, "classes_"):
            try:
                classes = list(ensemble.classes_)
            except Exception:
                classes = None

    return ensemble, scaler, classes


def extract_features(img_path, target_size=(224, 224), base_model=None):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feats = base_model.predict(arr, verbose=0)
    # ensure 1D vector
    feats = np.asarray(feats).reshape((feats.shape[0], -1))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for binary yes/no output (default 0.5)")
    args = p.parse_args()

    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(2)

    # Load artifacts
    try:
        ensemble, scaler, classes = load_artifacts()
    except Exception as e:
        print("Failed to load model artifacts:", e)
        sys.exit(3)

    # Build feature extractor
    base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

    feats = extract_features(img_path, base_model=base_model)
    X = scaler.transform(feats)

    # Get probabilities if available
    prob = None
    try:
        if hasattr(ensemble, "predict_proba"):
            prob = ensemble.predict_proba(X)[0]
    except Exception:
        prob = None

    if prob is None:
        # fallback to predict() and give confidence=1.0 for returned class
        pred_idx = ensemble.predict(X)[0]
        # If predict returns labels instead of indices, try to map
        if classes is not None and pred_idx in classes:
            label = pred_idx
            conf = 1.0
        else:
            # try to interpret as index
            try:
                pred_idx_int = int(pred_idx)
                label = classes[pred_idx_int] if classes is not None else str(pred_idx_int)
            except Exception:
                label = str(pred_idx)
            conf = 1.0
    else:
        idx = int(np.argmax(prob))
        conf = float(prob[idx])
        label = classes[idx] if classes is not None else str(idx)

    # Print the standard prediction info
    print(f"Image: {img_path}")
    print(f"Predicted: {label}")
    print(f"Confidence: {conf:.3f}")

    # Binary yes/no decision
    if conf >= args.threshold:
        print("Skin disease: YES")
    else:
        print("Skin disease: NO")


if __name__ == "__main__":
    main()
