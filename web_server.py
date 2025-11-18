"""Minimal Flask web server to upload an image and return prediction

Run:
  python3 web_server.py

Then open http://127.0.0.1:5000/ in a browser. The page lets you upload an image and shows the
predicted class, confidence, and a YES/NO decision.

Notes:
- This uses the existing `predict_single.py` helpers (load_artifacts, extract_features).
- Requires Flask to be installed: `pip install flask` (or add to your venv).
- Starting the server will load the ensemble and scaler from the project root and build
  an EfficientNetB0 feature extractor (this may take a few seconds on first run).
"""

from flask import Flask, render_template, request, jsonify
import os
import tempfile

app = Flask(__name__)

# Lazy imports to avoid blowing up at import time if TF isn't available
def init_models():
    # Import inside function to avoid import-time failures in environments without TF
    from predict_single import load_artifacts, extract_features as ps_extract_features
    from tensorflow.keras.applications.efficientnet import EfficientNetB0

    ensemble, scaler, classes = load_artifacts()
    base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    # also return the extract_features helper imported from predict_single
    return ensemble, scaler, classes, base_model, ps_extract_features

# Initialize on first request
_models = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global _models
    if _models is None:
        try:
            _models = init_models()
        except Exception as e:
            return jsonify({'error': f'Failed to load models: {e}'}), 500

    ensemble, scaler, classes, base_model, ps_extract_features = _models

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # save to a temporary file
    suffix = os.path.splitext(f.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        feats = ps_extract_features(tmp_path, base_model=base_model)
        X = scaler.transform(feats)

        prob = None
        try:
            if hasattr(ensemble, 'predict_proba'):
                prob = ensemble.predict_proba(X)[0]
        except Exception:
            prob = None

        if prob is None:
            pred_idx = ensemble.predict(X)[0]
            # map to class name if possible
            if classes is not None and pred_idx in classes:
                label = pred_idx
                conf = 1.0
            else:
                try:
                    pred_idx_int = int(pred_idx)
                    label = classes[pred_idx_int] if classes is not None else str(pred_idx_int)
                except Exception:
                    label = str(pred_idx)
                conf = 1.0
        else:
            idx = int(prob.argmax())
            conf = float(prob[idx])
            label = classes[idx] if classes is not None else str(idx)

        threshold = float(request.form.get('threshold', 0.5))
        decision = 'YES' if conf >= threshold else 'NO'

        result = {
            'predicted': label,
            'confidence': round(conf, 4),
            'disease': decision
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Failed during prediction: {e}'}), 500

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    # Simple dev server
    app.run(host='127.0.0.1', port=5000, debug=True)
