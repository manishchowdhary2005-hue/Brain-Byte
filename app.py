import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURATION ---
app = Flask(__name__)
MODEL_FILE = 'hemoscan_model.pkl'

# --- AI ENGINE: MODEL TRAINING LOGIC ---
def train_model():
    """Trains a Random Forest model using clinical logic for Anemia detection"""
    # Synthetic dataset generation based on WHO clinical ranges
    # Features: [Hemoglobin, RBC, Age, Gender(M:1, F:0), Fatigue(1/0), PaleSkin(1/0)]
    data = [
        [15.0, 5.0, 30, 1, 0, 0, 0], # Normal Male
        [13.5, 4.8, 25, 0, 0, 0, 0], # Normal Female
        [10.5, 3.8, 35, 1, 1, 0, 1], # Moderate Male
        [9.0, 3.2, 28, 0, 1, 1, 1],  # Moderate Female
        [6.5, 2.1, 45, 1, 1, 1, 2],  # High Risk Male
        [5.8, 1.9, 22, 0, 1, 1, 2],  # High Risk Female
        [11.5, 4.0, 30, 0, 0, 0, 1], # Mild/Moderate Female
        [14.2, 4.9, 50, 1, 0, 0, 0], # Normal
        [7.2, 2.5, 60, 0, 1, 1, 2],  # High Risk
    ]
    # Expand dataset slightly for better fitting
    df = pd.DataFrame(data * 20, columns=['hb', 'rbc', 'age', 'gender', 'fatigue', 'pale_skin', 'risk'])
    
    X = df.drop('risk', axis=1)
    y = df['risk']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

# Load or Train Model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = train_model()

# --- UI DESIGN (HTML/CSS) ---
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HemoScan AI | Anemia Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #f0f4f8; font-family: 'Inter', sans-serif; color: #2d3436; }
        .navbar { background: linear-gradient(135deg, #004d40 0%, #00796b 100%); padding: 1rem; }
        .main-card { border: none; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); background: white; }
        .btn-analyze { background: #00796b; border: none; padding: 12px; font-weight: 600; border-radius: 10px; transition: 0.3s; }
        .btn-analyze:hover { background: #004d40; transform: translateY(-2px); }
        .form-label { font-weight: 600; font-size: 0.9rem; color: #495057; }
        .risk-badge { padding: 15px; border-radius: 12px; font-weight: bold; font-size: 1.5rem; text-align: center; margin: 20px 0; }
        .low-risk { background-color: #d1e7dd; color: #0f5132; }
        .mod-risk { background-color: #fff3cd; color: #856404; }
        .high-risk { background-color: #f8d7da; color: #842029; }
        .medical-icon { font-size: 2rem; margin-bottom: 10px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark shadow-sm">
        <div class="container">
            <a class="navbar-brand fw-bold" href="/">🩸 HEMOSCAN AI <span class="badge bg-light text-dark ms-2" style="font-size: 0.6rem;">SMART HEALTH</span></a>
        </div>
    </nav>

    <div class="container my-5">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                {% if not prediction %}
                <!-- INPUT FORM -->
                <div class="card main-card p-4 p-md-5">
                    <div class="text-center mb-4">
                        <div class="medical-icon">🔬</div>
                        <h2 class="fw-bold">Early Detection Scan</h2>
                        <p class="text-muted">Fill in patient vitals for automated risk analysis</p>
                    </div>
                    
                    <form action="/analyze" method="POST">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Hemoglobin (g/dL)</label>
                                <input type="number" step="0.1" name="hb" class="form-control" placeholder="13.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">RBC Count (m/µL)</label>
                                <input type="number" step="0.1" name="rbc" class="form-control" placeholder="4.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Age</label>
                                <input type="number" name="age" class="form-control" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Gender</label>
                                <select name="gender" class="form-select">
                                    <option value="1">Male</option>
                                    <option value="0">Female</option>
                                </select>
                            </div>
                            <div class="col-12 mt-4">
                                <h6 class="fw-bold">Observed Symptoms</h6>
                                <div class="form-check form-check-inline">
                                    <input class="form-check-input" type="checkbox" name="fatigue" value="1">
                                    <label class="form-check-label">Fatigue</label>
                                </div>
                                <div class="form-check form-check-inline">
                                    <input class="form-check-input" type="checkbox" name="pale" value="1">
                                    <label class="form-check-label">Pale Skin</label>
                                </div>
                            </div>
                            <div class="col-12 mt-4">
                                <button type="submit" class="btn btn-analyze btn-primary w-100">RUN AI DIAGNOSIS</button>
                            </div>
                        </div>
                    </form>
                </div>
                {% else %}
                <!-- RESULT SCREEN -->
                <div class="card main-card p-4 p-md-5 animate__animated animate__fadeIn">
                    <h3 class="text-center fw-bold">Analysis Report</h3>
                    <hr>
                    <div class="risk-badge {{ css_class }}">
                        {{ prediction }}
                    </div>
                    
                    <div class="row text-center mt-4">
                        <div class="col-4 border-end">
                            <p class="text-muted small mb-0">Hemoglobin</p>
                            <h5 class="fw-bold">{{ hb }}</h5>
                        </div>
                        <div class="col-4 border-end">
                            <p class="text-muted small mb-0">RBC</p>
                            <h5 class="fw-bold">{{ rbc }}</h5>
                        </div>
                        <div class="col-4">
                            <p class="text-muted small mb-0">Status</p>
                            <h5 class="fw-bold">{{ "Critical" if prediction == "High Risk" else "Stable" }}</h5>
                        </div>
                    </div>

                    <div class="alert alert-secondary mt-4">
                        <strong>AI Recommendation:</strong> 
                        {% if prediction == "High Risk" %}
                        Urgent consultation with a Hematologist and Iron studies required.
                        {% elif prediction == "Moderate Risk" %}
                        Dietary adjustments and Vitamin B12 / Iron supplements recommended.
                        {% else %}
                        Maintain a balanced diet. Routine checkup in 6 months.
                        {% endif %}
                    </div>
                    
                    <a href="/" class="btn btn-outline-secondary w-100 mt-2">New Assessment</a>
                </div>
                {% endif %}
            </div>
        </div>
        <footer class="text-center mt-5 text-muted small">
            &copy; 2024 HemoScan AI System | Intelligent Healthcare Solutions
        </footer>
    </div>
</body>
</html>
"""

# --- BACKEND ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_LAYOUT)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Extract and process inputs
        hb = float(request.form.get('hb', 0))
        rbc = float(request.form.get('rbc', 0))
        age = int(request.form.get('age', 0))
        gender = int(request.form.get('gender', 1))
        fatigue = 1 if request.form.get('fatigue') else 0
        pale = 1 if request.form.get('pale') else 0
        
        # Prepare data for AI model
        input_data = np.array([[hb, rbc, age, gender, fatigue, pale]])
        prediction_id = model.predict(input_data)[0]
        
        # Map prediction to UI
        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        class_map = {0: "low-risk", 1: "mod-risk", 2: "high-risk"}
        
        return render_template_string(
            HTML_LAYOUT, 
            prediction=risk_map[prediction_id],
            css_class=class_map[prediction_id],
            hb=hb, 
            rbc=rbc
        )
    except Exception as e:
        return f"Error in processing: {str(e)}"

if __name__ == '__main__':
    # Threaded mode for smooth handling
    app.run(host='0.0.0.0', port=5000, debug=True)
