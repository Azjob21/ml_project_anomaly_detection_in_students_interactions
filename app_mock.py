"""
Student Anomaly Detection API - Mock Version
Use this to test deployment without trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

print("="*60)
print("🚀 RUNNING IN MOCK MODE")
print("Using rule-based predictions instead of ML model")
print("="*60)


def calculate_risk_score(student_data):
    """
    Calculate risk score using simple rules
    (Replace with real model when available)
    """
    score = 0.0
    
    # Academic performance (40% weight)
    avg_score = student_data.get('avg_score', 50)
    if avg_score < 30:
        score += 0.4
    elif avg_score < 40:
        score += 0.3
    elif avg_score < 50:
        score += 0.2
    elif avg_score < 60:
        score += 0.1
    
    # Engagement (30% weight)
    total_clicks = student_data.get('total_clicks', 500)
    if total_clicks < 200:
        score += 0.3
    elif total_clicks < 400:
        score += 0.2
    elif total_clicks < 600:
        score += 0.1
    
    # Assessments (15% weight)
    num_assessments = student_data.get('num_assessments', 5)
    if num_assessments < 3:
        score += 0.15
    elif num_assessments < 5:
        score += 0.1
    
    # Previous attempts (10% weight)
    prev_attempts = student_data.get('num_of_prev_attempts', 0)
    if prev_attempts > 0:
        score += min(0.1 * prev_attempts, 0.2)
    
    # Late submissions (5% weight)
    avg_submission = student_data.get('avg_submission_date', 100)
    if avg_submission > 180:
        score += 0.05
    elif avg_submission > 150:
        score += 0.03
    
    # Cheating detection (bonus)
    if avg_score > 85 and total_clicks < 250:
        score += 0.2  # High score with low engagement = suspicious
    
    return min(score, 1.0)  # Cap at 1.0


def calculate_risk_factors(student_data):
    """
    Identify key risk factors
    """
    factors = []
    
    avg_score = student_data.get('avg_score', 50)
    total_clicks = student_data.get('total_clicks', 500)
    num_assessments = student_data.get('num_assessments', 5)
    prev_attempts = student_data.get('num_of_prev_attempts', 0)
    avg_submission = student_data.get('avg_submission_date', 100)
    num_interactions = student_data.get('num_interactions', 10)
    
    # Check for cheating patterns
    if avg_score > 80 and total_clicks < 200:
        factors.append({
            'name': 'High Score with Low Engagement (Possible Cheating)',
            'impact': 'high',
            'value': f"Score: {avg_score}, Clicks: {total_clicks}"
        })
    
    # Low academic performance
    if avg_score < 40:
        factors.append({
            'name': 'Very Low Average Score',
            'impact': 'high',
            'value': avg_score
        })
    elif avg_score < 50:
        factors.append({
            'name': 'Low Average Score',
            'impact': 'medium',
            'value': avg_score
        })
    
    # Low engagement
    if total_clicks < 300:
        factors.append({
            'name': 'Low Platform Engagement',
            'impact': 'high',
            'value': total_clicks
        })
    elif total_clicks < 500:
        factors.append({
            'name': 'Below Average Engagement',
            'impact': 'medium',
            'value': total_clicks
        })
    
    # Few assessments
    if num_assessments < 4:
        factors.append({
            'name': 'Few Assessments Completed',
            'impact': 'medium',
            'value': num_assessments
        })
    
    # Previous attempts
    if prev_attempts > 0:
        factors.append({
            'name': 'Previous Course Attempts',
            'impact': 'medium',
            'value': prev_attempts
        })
    
    # Late submissions
    if avg_submission > 150:
        factors.append({
            'name': 'Late Assignment Submissions',
            'impact': 'medium',
            'value': avg_submission
        })
    
    # Low interactions
    if num_interactions < 8:
        factors.append({
            'name': 'Low Learning Activity',
            'impact': 'medium',
            'value': num_interactions
        })
    
    return factors[:5]  # Top 5 factors


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Student Anomaly Detection API (Mock Mode)',
        'version': '1.0-mock',
        'model_loaded': True,
        'note': 'Using rule-based predictions for testing'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a student is at risk
    """
    try:
        student_data = request.json
        
        if not student_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Calculate risk score
        risk_score = calculate_risk_score(student_data)
        
        # Determine if anomaly
        is_anomaly = risk_score > 0.4
        
        # Get risk factors
        risk_factors = calculate_risk_factors(student_data)
        
        # Prepare response
        response = {
            'isAnomaly': is_anomaly,
            'riskScore': float(risk_score),
            'confidence': 0.85,  # Mock confidence
            'factors': risk_factors,
            'recommendation': 'Immediate intervention recommended' if is_anomaly else 'Continue monitoring',
            'mode': 'mock'  # Indicator that this is mock mode
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict for multiple students
    """
    try:
        data = request.json
        students = data.get('students', [])
        
        if not students:
            return jsonify({'error': 'No students data provided'}), 400
        
        results = []
        for student in students:
            risk_score = calculate_risk_score(student)
            is_anomaly = risk_score > 0.4
            
            results.append({
                'student_id': student.get('id_student', 'unknown'),
                'isAnomaly': is_anomaly,
                'riskScore': float(risk_score)
            })
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'mode': 'mock'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Rule-Based (Mock)',
        'accuracy': 0.85,
        'mode': 'mock',
        'note': 'Replace with real model for production',
        'features_used': [
            'avg_score',
            'total_clicks',
            'num_assessments',
            'num_of_prev_attempts',
            'avg_submission_date',
            'num_interactions'
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'mode': 'mock'
    })


if __name__ == '__main__':
    print("\n📝 NOTE: This is a MOCK version for testing")
    print("👉 To use the real ML model, save your trained models and use app.py\n")
    app.run(host='0.0.0.0', port=5000, debug=True)