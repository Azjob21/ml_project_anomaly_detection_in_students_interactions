import pytest
import sys
import os
import json

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test data samples
AT_RISK_STUDENT = {
    "code_module": "AAA",
    "code_presentation": "2013J",
    "gender": "M",
    "region": "East Anglian Region",
    "highest_education": "Lower Than A Level",
    "imd_band": "20-30%",
    "age_band": "35-55",
    "disability": "N",
    "studied_credits": 60,
    "num_of_prev_attempts": 2,
    "avg_score": 35.5,
    "std_score": 12.2,
    "min_score": 15,
    "max_score": 55,
    "num_assessments": 3,
    "avg_submission_date": 180,
    "std_submission_date": 40,
    "score_range": 40,
    "total_clicks": 150,
    "avg_clicks": 20,
    "std_clicks": 15,
    "max_clicks": 80,
    "num_interactions": 5,
    "first_access": 100,
    "last_access": 200,
    "access_duration": 100,
    "avg_registration_date": -10,
    "num_unregistrations": 1
}

NORMAL_STUDENT = {
    "code_module": "BBB",
    "code_presentation": "2013J",
    "gender": "F",
    "region": "Scotland",
    "highest_education": "HE Qualification",
    "imd_band": "30-40%",
    "age_band": "0-35",
    "disability": "N",
    "studied_credits": 60,
    "num_of_prev_attempts": 0,
    "avg_score": 75.5,
    "std_score": 10.2,
    "min_score": 60,
    "max_score": 95,
    "num_assessments": 8,
    "avg_submission_date": 90,
    "std_submission_date": 20,
    "score_range": 35,
    "total_clicks": 850,
    "avg_clicks": 100,
    "std_clicks": 30,
    "max_clicks": 200,
    "num_interactions": 15,
    "first_access": 10,
    "last_access": 200,
    "access_duration": 190,
    "avg_registration_date": -20,
    "num_unregistrations": 0
}

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/')
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = json.loads(response.data)
    assert data['status'] == 'running'
    assert 'version' in data

def test_model_info(client):
    """Test the model info endpoint"""
    response = client.get('/info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_type' in data
    assert 'version' in data

def test_prediction_at_risk(client):
    """Test prediction for at-risk student"""
    response = client.post('/predict', 
                          data=json.dumps(AT_RISK_STUDENT),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['isAtRisk'] is True
    assert data['riskScore'] > 50

def test_prediction_normal(client):
    """Test prediction for normal student"""
    response = client.post('/predict', 
                          data=json.dumps(NORMAL_STUDENT),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['isAtRisk'] is False
    assert data['riskScore'] <= 50

def test_batch_prediction(client):
    """Test batch prediction endpoint"""
    batch_data = {
        "students": [
            {**AT_RISK_STUDENT, "id_student": 1001},
            {**NORMAL_STUDENT, "id_student": 1002}
        ]
    }
    
    response = client.post('/predict_batch', 
                          data=json.dumps(batch_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'summary' in data
    assert 'predictions' in data
    assert len(data['predictions']) == 2
    assert data['summary']['total_students'] == 2

def test_error_handling(client):
    """Test error handling with invalid data"""
    invalid_data = {"avg_score": 50}  # Missing most fields
    
    response = client.post('/predict', 
                          data=json.dumps(invalid_data),
                          content_type='application/json')
    
    # Depending on implementation, might return 200 with default values or 400/500
    # The original implementation seems to handle missing keys with defaults in `preprocess_input`
    # or fail if `data` is empty. Let's check `preprocess_input`.
    # It fills missing cols with 0. So it should actually succeed but maybe predict weirdly.
    # Wait, `preprocess_input` fills missing cols with 0.
    
    # However, `calculate_risk_score_advanced` might fail if keys are missing?
    # No, it uses `.get()` with defaults.
    
    # So actually, invalid input (JSON object) IS processed.
    # But invalid JSON (malformed) would be 400 from Flask.
    
    # Let's test empty input
    response = client.post('/predict', 
                          data=json.dumps({}), # Empty dict
                          content_type='application/json')
    assert response.status_code == 400 # "No data provided" check in app.py