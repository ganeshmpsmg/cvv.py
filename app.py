import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tempfile
import base64
from io import BytesIO
from PIL import Image as PILImage

app = Flask(__name__)
CORS(app)  # Enable CORS for Netlify frontend
app.secret_key = "secret123"

print("🔹 Loading pretrained MobileNetV2 model (ImageNet)...")
model = MobileNetV2(weights='imagenet')
print("✅ Model loaded successfully!")

# In-memory session storage (for demo purposes)
active_sessions = {}

# --------------------------
# ROOT ENDPOINT
# --------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Image Classification API with Authentication",
        "endpoints": {
            "/api/login": "POST - Login with username and password",
            "/api/validate-session": "POST - Validate user session",
            "/api/classify": "POST - Upload image for classification",
            "/api/logout": "POST - Logout user"
        }
    })


# --------------------------
# LOGIN ENDPOINT
# --------------------------
@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username", "")
        password = data.get("password", "")
        
        # Validate credentials (same as original code)
        if username == "Ganesh" and password == "1234":
            # Create session token (simple implementation)
            import secrets
            session_token = secrets.token_hex(16)
            active_sessions[session_token] = username
            
            return jsonify({
                "success": True,
                "message": "Login successful",
                "username": username,
                "session_token": session_token
            })
        else:
            return jsonify({
                "success": False,
                "error": "Invalid username or password"
            }), 401
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------
# VALIDATE SESSION
# --------------------------
@app.route("/api/validate-session", methods=["POST"])
def validate_session():
    try:
        data = request.get_json()
        session_token = data.get("session_token", "")
        
        if session_token in active_sessions:
            return jsonify({
                "success": True,
                "username": active_sessions[session_token]
            })
        else:
            return jsonify({
                "success": False,
                "error": "Invalid or expired session"
            }), 401
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------
# IMAGE CLASSIFICATION ENDPOINT
# --------------------------
@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        # Check for file upload
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Load and preprocess image (same as original code)
        img = image.load_img(tmp_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Predict (same as original code)
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=1)[0][0]
        
        label = decoded[1]
        confidence = round(decoded[2] * 100, 2)
        
        # Read image and convert to base64 for frontend display
        with open(tmp_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence,
            "image_data": f"data:image/jpeg;base64,{img_data}",
            "filename": file.filename
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------
# LOGOUT ENDPOINT
# --------------------------
@app.route("/api/logout", methods=["POST"])
def logout():
    try:
        data = request.get_json()
        session_token = data.get("session_token", "")
        
        if session_token in active_sessions:
            del active_sessions[session_token]
        
        return jsonify({
            "success": True,
            "message": "Logged out successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------
# ABOUT INFO ENDPOINT
# --------------------------
@app.route("/api/about", methods=["GET"])
def about():
    return jsonify({
        "success": True,
        "app_name": "Image Classification System",
        "model": "MobileNetV2",
        "dataset": "ImageNet",
        "description": "This application uses a pre-trained MobileNetV2 deep learning model trained on the ImageNet dataset to classify images into 1000+ categories.",
        "developer": "Ganesh",
        "version": "1.0.0"
    })


# --------------------------
# HEALTH CHECK
# --------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "active_sessions": len(active_sessions)
    })


# --------------------------
# RUN
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)