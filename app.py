"""
Student Anomaly Detection API - Fixed Prediction Logic
Flask backend for serving the trained model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)

# Fix CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables for models
model = None
scaler = None
label_encoders = None

def load_models():
    """Load the trained models from the models directory"""
    global model, scaler, label_encoders
    
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        if not os.path.exists(models_dir):
            print(f"⚠️  Models directory not found: {models_dir}")
            return False
        
        model_path = os.path.join(models_dir, 'best_anomaly_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
        
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        print(f"Loading encoders from: {encoders_path}")
        
        for path in [model_path, scaler_path, encoders_path]:
            if not os.path.exists(path):
                print(f"⚠️  Model file not found: {path}")
                return False
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        
        print("="*60)
        print("✓ Successfully loaded all models!")
        print(f"Model type: {type(model).__name__}")
        if hasattr(model, 'contamination'):
            print(f"Model contamination: {model.contamination}")
        if hasattr(model, 'threshold_'):
            print(f"Model threshold: {model.threshold_}")
        print("="*60)
        return True
        
    except Exception as e:
        print("="*60)
        print(f"✗ Error loading models: {e}")
        traceback.print_exc()
        print("="*60)
        return False

def preprocess_input(data):
    """Preprocess the input data for prediction"""
    try:
        # Create DataFrame
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        
        # Encode categorical variables
        categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                           'highest_education', 'imd_band', 'age_band', 'disability']
        
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                try:
                    df[col + '_encoded'] = le.transform(df[col].astype(str))
                except ValueError as e:
                    print(f"⚠️  Unknown value for {col}: using default")
                    df[col + '_encoded'] = 0
            else:
                df[col + '_encoded'] = 0
        
        # Define all feature columns (must match training order)
        feature_cols = [col + '_encoded' for col in categorical_cols] + [
            'studied_credits', 'num_of_prev_attempts',
            'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
            'avg_submission_date', 'std_submission_date', 'score_range',
            'total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks',
            'num_interactions', 'first_access', 'last_access', 'access_duration',
            'avg_registration_date', 'num_unregistrations'
        ]
        
        # Ensure all features exist with default values
        for feature in feature_cols:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[feature_cols]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        traceback.print_exc()
        raise ValueError(f"Error preprocessing input: {str(e)}")

def calculate_risk_score_advanced(raw_score, prediction, student_data):
    """
    Advanced risk calculation that uses BOTH anomaly score AND actual student metrics
    This prevents all students from getting the same prediction
    """
    
    # Start with anomaly score based risk
    if raw_score >= -0.45:
        base_risk = 0
    elif raw_score >= -0.50:
        base_risk = 10
    elif raw_score >= -0.55:
        base_risk = 25
    elif raw_score >= -0.60:
        base_risk = 40
    elif raw_score >= -0.70:
        base_risk = 55
    elif raw_score >= -0.80:
        base_risk = 70
    elif raw_score >= -1.0:
        base_risk = 85
    else:
        base_risk = 95
    
    # Add rule-based adjustments based on actual metrics
    risk_adjustment = 0
    
    avg_score = student_data.get('avg_score', 50)
    total_clicks = student_data.get('total_clicks', 500)
    num_assessments = student_data.get('num_assessments', 5)
    num_interactions = student_data.get('num_interactions', 10)
    prev_attempts = student_data.get('num_of_prev_attempts', 0)
    
    # Academic performance adjustment
    if avg_score < 30:
        risk_adjustment += 25
    elif avg_score < 40:
        risk_adjustment += 15
    elif avg_score < 50:
        risk_adjustment += 8
    elif avg_score > 85:
        risk_adjustment -= 15
    elif avg_score > 75:
        risk_adjustment -= 10
    
    # Engagement adjustment
    if total_clicks < 200:
        risk_adjustment += 20
    elif total_clicks < 400:
        risk_adjustment += 12
    elif total_clicks < 600:
        risk_adjustment += 5
    elif total_clicks > 1200:
        risk_adjustment -= 15
    elif total_clicks > 900:
        risk_adjustment -= 10
    
    # Assessments completed
    if num_assessments < 3:
        risk_adjustment += 12
    elif num_assessments < 5:
        risk_adjustment += 5
    elif num_assessments >= 8:
        risk_adjustment -= 8
    
    # Interactions
    if num_interactions < 5:
        risk_adjustment += 10
    elif num_interactions < 8:
        risk_adjustment += 5
    elif num_interactions >= 15:
        risk_adjustment -= 8
    
    # Previous attempts
    if prev_attempts >= 3:
        risk_adjustment += 15
    elif prev_attempts >= 2:
        risk_adjustment += 10
    elif prev_attempts >= 1:
        risk_adjustment += 5
    
    # Check for suspicious patterns (cheating)
    if avg_score > 85 and total_clicks < 300:
        risk_adjustment += 20  # Suspicious pattern
    
    # Calculate final risk
    final_risk = max(0, min(100, base_risk + risk_adjustment))
    
    # Override: If model says normal AND metrics are good, keep low risk
    if prediction == 1 and avg_score > 70 and total_clicks > 800:
        final_risk = min(final_risk, 30)
    
    # Override: If model says anomaly AND metrics are bad, ensure high risk
    if prediction == -1 and (avg_score < 40 or total_clicks < 300):
        final_risk = max(final_risk, 60)
    
    return final_risk

def analyze_risk_factors(data, is_at_risk, risk_score):
    """Analyze which factors contribute to risk"""
    factors = []
    
    avg_score = data.get('avg_score', 50)
    total_clicks = data.get('total_clicks', 0)
    num_assessments = data.get('num_assessments', 0)
    prev_attempts = data.get('num_of_prev_attempts', 0)
    avg_submission = data.get('avg_submission_date', 0)
    num_interactions = data.get('num_interactions', 0)
    first_access = data.get('first_access', 0)
    
    if avg_score < 40:
        factors.append({
            'factor': 'Very Low Average Score',
            'value': avg_score,
            'severity': 'high',
            'description': f'Average score of {avg_score:.1f}% is significantly below passing'
        })
    elif avg_score < 60:
        factors.append({
            'factor': 'Low Average Score',
            'value': avg_score,
            'severity': 'medium',
            'description': f'Average score of {avg_score:.1f}% indicates struggling'
        })
    
    if total_clicks < 200:
        factors.append({
            'factor': 'Very Low Platform Engagement',
            'value': total_clicks,
            'severity': 'high',
            'description': f'Only {total_clicks} total clicks shows minimal engagement'
        })
    elif total_clicks < 500:
        factors.append({
            'factor': 'Low Platform Engagement',
            'value': total_clicks,
            'severity': 'medium',
            'description': f'{total_clicks} clicks is below average engagement'
        })
    
    if num_assessments < 3:
        factors.append({
            'factor': 'Few Assessments Completed',
            'value': num_assessments,
            'severity': 'high' if num_assessments < 2 else 'medium',
            'description': f'Only {num_assessments} assessments completed'
        })
    
    if prev_attempts > 1:
        factors.append({
            'factor': 'Multiple Previous Attempts',
            'value': prev_attempts,
            'severity': 'medium',
            'description': f'{prev_attempts} previous attempts indicates persistent difficulty'
        })
    
    if avg_submission > 150:
        factors.append({
            'factor': 'Consistently Late Submissions',
            'value': avg_submission,
            'severity': 'medium',
            'description': 'Assignments submitted late on average'
        })
    
    if num_interactions < 5:
        factors.append({
            'factor': 'Very Low Learning Activity',
            'value': num_interactions,
            'severity': 'high',
            'description': f'Only {num_interactions} learning interactions recorded'
        })
    
    # CHEATING DETECTION
    if avg_score > 85 and total_clicks < 300:
        factors.append({
            'factor': 'Suspicious Pattern: High Scores with Low Engagement',
            'value': f"Score: {avg_score:.1f}%, Clicks: {total_clicks}",
            'severity': 'high',
            'description': 'Unusually high performance with minimal platform use may warrant investigation'
        })
    
    if first_access > 50:
        factors.append({
            'factor': 'Late Course Start',
            'value': first_access,
            'severity': 'medium',
            'description': f'First accessed course on day {first_access}'
        })
    
    return factors

def generate_recommendation(is_at_risk, risk_score, risk_factors):
    """Generate intervention recommendations"""
    if risk_score < 30:
        return "Student appears to be performing well. Continue standard monitoring."
    
    if risk_score > 70:
        high_severity = [f for f in risk_factors if f.get('severity') == 'high']
        if high_severity:
            main_issues = ', '.join([f['factor'] for f in high_severity[:2]])
            return f"⚠️ Immediate intervention recommended. Primary concerns: {main_issues}"
        return "⚠️ Immediate intervention recommended. Student showing multiple risk factors."
    
    if risk_score > 50:
        return "Schedule check-in with student within 1 week to address concerns."
    
    return "Monitor closely and consider reaching out to offer support."

@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'running',
        'message': 'Student Anomaly Detection API',
        'version': '1.3 - Fixed Risk Calculation',
        'model_loaded': model is not None,
        'endpoints': {
            '/predict': 'Single student prediction',
            '/predict_batch': 'Batch CSV prediction',
            '/diagnose': 'Test prediction on sample data'
        }
    })

@app.route('/diagnose', methods=['GET'])
def diagnose():
    """Diagnostic endpoint to test model behavior"""
    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    # Create test samples
    test_samples = [
        {'name': 'Excellent Student', 'avg_score': 90, 'total_clicks': 1500, 'num_assessments': 10},
        {'name': 'Average Student', 'avg_score': 65, 'total_clicks': 700, 'num_assessments': 7},
        {'name': 'Struggling Student', 'avg_score': 35, 'total_clicks': 250, 'num_assessments': 3},
        {'name': 'Suspicious Pattern', 'avg_score': 95, 'total_clicks': 150, 'num_assessments': 5}
    ]
    
    results = []
    for sample in test_samples:
        test_data = {
            'code_module': 'AAA', 'code_presentation': '2013J', 'gender': 'M',
            'region': 'East Anglian Region', 'highest_education': 'HE Qualification',
            'imd_band': '20-30%', 'age_band': '35-55', 'disability': 'N',
            'studied_credits': 60, 'num_of_prev_attempts': 0,
            'avg_score': sample['avg_score'], 'std_score': 15, 'min_score': 20,
            'max_score': 85, 'num_assessments': sample['num_assessments'],
            'avg_submission_date': 100, 'std_submission_date': 30, 'score_range': 65,
            'total_clicks': sample['total_clicks'], 'avg_clicks': 50, 'std_clicks': 20,
            'max_clicks': 150, 'num_interactions': 10, 'first_access': 10,
            'last_access': 200, 'access_duration': 190, 'avg_registration_date': -15,
            'num_unregistrations': 0
        }
        
        X = preprocess_input(test_data)
        pred = model.predict(X)[0]
        score = model.score_samples(X)[0]
        risk = calculate_risk_score_advanced(score, pred, test_data)
        
        results.append({
            'student': sample['name'],
            'prediction': int(pred),
            'anomaly_score': float(score),
            'risk_score': float(risk),
            'interpretation': 'Anomaly' if pred == -1 else 'Normal'
        })
    
    return jsonify({
        'diagnosis': results,
        'model_info': {
            'type': type(model).__name__,
            'contamination': getattr(model, 'contamination', 'N/A')
        }
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict if a student is at risk"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        X = preprocess_input(data)
        prediction = model.predict(X)[0]
        raw_score = model.score_samples(X)[0]
        
        # Use advanced risk calculation
        risk_score = calculate_risk_score_advanced(raw_score, prediction, data)
        is_at_risk = risk_score >= 50  # Risk-based threshold instead of model prediction
        
        risk_factors = analyze_risk_factors(data, is_at_risk, risk_score)
        recommendation = generate_recommendation(is_at_risk, risk_score, risk_factors)
        
        print(f"✓ Prediction: pred={prediction}, score={raw_score:.3f}, risk={risk_score:.1f}%")
        
        return jsonify({
            'isAtRisk': bool(is_at_risk),
            'riskScore': float(risk_score),
            'anomalyScore': float(raw_score),
            'prediction': int(prediction),
            'confidence': 0.85,
            'riskFactors': risk_factors,
            'recommendation': recommendation
        })
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST', 'OPTIONS'])
def predict_batch():
    """Predict risk for multiple students from CSV data"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        if not data or 'students' not in data:
            return jsonify({'error': 'No student data provided'}), 400
        
        students_data = data['students']
        print(f"📥 Batch prediction for {len(students_data)} students")
        
        df = pd.DataFrame(students_data)
        student_ids = df.get('student_id', range(len(df))).tolist()
        
        X = preprocess_input(df)
        predictions = model.predict(X)
        raw_scores = model.score_samples(X)
        
        results = []
        for idx, (pred, raw_score) in enumerate(zip(predictions, raw_scores)):
            student_data = students_data[idx]
            risk_score = calculate_risk_score_advanced(raw_score, pred, student_data)
            is_at_risk = risk_score >= 50
            
            risk_factors = analyze_risk_factors(student_data, is_at_risk, risk_score)
            recommendation = generate_recommendation(is_at_risk, risk_score, risk_factors)
            
            results.append({
                'student_id': student_ids[idx],
                'isAtRisk': bool(is_at_risk),
                'riskScore': float(risk_score),
                'anomalyScore': float(raw_score),
                'prediction': int(pred),
                'confidence': 0.85,
                'riskLevel': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low',
                'numRiskFactors': len(risk_factors),
                'topRiskFactors': [f['factor'] for f in risk_factors[:3]],
                'recommendation': recommendation,
                'avg_score': student_data.get('avg_score', 0),
                'total_clicks': student_data.get('total_clicks', 0),
                'num_assessments': student_data.get('num_assessments', 0)
            })
        
        total = len(results)
        at_risk = sum(1 for r in results if r['isAtRisk'])
        high_risk = sum(1 for r in results if r['riskScore'] > 70)
        medium_risk = sum(1 for r in results if 40 < r['riskScore'] <= 70)
        low_risk = sum(1 for r in results if r['riskScore'] <= 40)
        
        summary = {
            'total_students': total,
            'at_risk_count': at_risk,
            'at_risk_percentage': round((at_risk / total) * 100, 1),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'average_risk_score': round(sum(r['riskScore'] for r in results) / total, 1)
        }
        
        print(f"✓ Batch complete: {at_risk}/{total} at-risk ({summary['at_risk_percentage']}%)")
        
        return jsonify({
            'summary': summary,
            'predictions': results
        })
        
    except Exception as e:
        print(f"✗ Batch error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/info', methods=['GET', 'OPTIONS'])
def model_info():
    """Get information about the model"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'model_type': 'Isolation Forest',
        'model_loaded': model is not None,
        'version': '1.3',
        'description': 'Hybrid approach using ML anomaly detection + rule-based risk assessment'
    })

# Load models on startup
print("\n" + "="*60)
print("🚀 Starting Student Anomaly Detection API v1.3")
print("="*60)
models_loaded = load_models()

if not models_loaded:
    print("\n⚠️  WARNING: Models not loaded!")

if __name__ == '__main__':
    print("\n🌐 Server starting on http://0.0.0.0:5000")
    print("📊 Visit /diagnose to test model predictions")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)