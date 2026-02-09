import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ML components BEFORE importing app to handle module-level code if needed
# although app.py loads them at the end.
mock_model = MagicMock()
mock_scaler = MagicMock()
mock_encoders = {}

# Configure mocks to return valid types for comparison/array indexing
mock_model.predict.return_value = np.array([1])
mock_model.score_samples.return_value = np.array([-0.4])
mock_scaler.transform.side_effect = lambda x: np.array(x)

# Setup common categorical columns for label encoders
categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                   'highest_education', 'imd_band', 'age_band', 'disability']
for col in categorical_cols:
    le = MagicMock()
    # Mock transform to return a numpy array of 0s with same length as input
    le.transform.side_effect = lambda x: np.zeros(len(x), dtype=int)
    mock_encoders[col] = le

@pytest.fixture(autouse=True)
def setup_mocks():
    """Apply mocks to the app module globals"""
    with patch('app.model', mock_model), \
         patch('app.scaler', mock_scaler), \
         patch('app.label_encoders', mock_encoders), \
         patch('app.load_models', return_value=True):
        yield

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
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'running'

def test_model_info(client):
    """Test the model info endpoint"""
    response = client.get('/info')
    assert response.status_code == 200
    data = response.get_json()
    assert 'model_type' in data

def test_prediction_at_risk(client):
    """Test prediction for at-risk student"""
    # Force anomaly prediction
    mock_model.predict.return_value = np.array([-1])
    mock_model.score_samples.return_value = np.array([-0.8])
    
    response = client.post('/predict', 
                          data=json.dumps(AT_RISK_STUDENT),
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['riskScore'] >= 50

def test_prediction_normal(client):
    """Test prediction for normal student"""
    # Force normal prediction
    mock_model.predict.return_value = np.array([1])
    mock_model.score_samples.return_value = np.array([-0.3])
    
    response = client.post('/predict', 
                          data=json.dumps(NORMAL_STUDENT),
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['riskScore'] < 50

def test_batch_prediction(client):
    """Test batch prediction endpoint"""
    batch_data = {
        "students": [AT_RISK_STUDENT, NORMAL_STUDENT]
    }
    
    # Mock for batch (2 students)
    mock_model.predict.return_value = np.array([-1, 1])
    mock_model.score_samples.return_value = np.array([-0.8, -0.3])
    
    response = client.post('/predict_batch', 
                          data=json.dumps(batch_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['predictions']) == 2

def test_error_handling(client):
    """Test error handling with empty input"""
    response = client.post('/predict', 
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400