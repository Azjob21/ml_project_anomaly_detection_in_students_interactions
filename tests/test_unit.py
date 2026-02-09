import pytest
import sys
import os

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import calculate_risk_score_advanced, analyze_risk_factors

def test_risk_calculation_high_risk():
    """Test that a student with bad metrics gets a high risk score"""
    prediction = -1  # Anomaly
    raw_score = -0.75
    student_data = {
        'avg_score': 35,
        'total_clicks': 150,
        'num_assessments': 2,
        'num_interactions': 3,
        'num_of_prev_attempts': 2
    }
    
    risk = calculate_risk_score_advanced(raw_score, prediction, student_data)
    assert risk >= 50, f"Expected high risk, got {risk}"

def test_risk_calculation_low_risk():
    """Test that a student with good metrics gets a low risk score"""
    prediction = 1  # Normal
    raw_score = -0.40
    student_data = {
        'avg_score': 85,
        'total_clicks': 1000,
        'num_assessments': 10,
        'num_interactions': 20,
        'num_of_prev_attempts': 0
    }
    
    risk = calculate_risk_score_advanced(raw_score, prediction, student_data)
    assert risk < 30, f"Expected low risk, got {risk}"

def test_risk_factors_identification():
    """Test that risk factors are correctly identified"""
    data = {
        'avg_score': 30,
        'total_clicks': 100
    }
    
    factors = analyze_risk_factors(data, True, 80)
    factor_names = [f['factor'] for f in factors]
    
    assert 'Very Low Average Score' in factor_names
    assert 'Very Low Platform Engagement' in factor_names
