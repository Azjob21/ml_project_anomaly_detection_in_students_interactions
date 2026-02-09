"""
API Test Script - Compatible with Standard Flask API
"""

import requests
import json

API_URL = "http://localhost:5000"

print("="*60)
print("🧪 TESTING STUDENT ANOMALY DETECTION API")
print("="*60)

# Test 1: Health Check
print("\n[TEST 1] Health Check...")
try:
    response = requests.get(f"{API_URL}/")
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"✗ Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Normal Student
print("\n[TEST 2] Normal Student Prediction...")
normal_student = {
    "code_module": "AAA",
    "code_presentation": "2013J",
    "gender": "F",
    "region": "Scotland",
    "highest_education": "HE Qualification",
    "imd_band": "30-40%",
    "age_band": "0-35",
    "disability": "N",
    "studied_credits": 60,
    "num_of_prev_attempts": 0,
    "avg_score": 75.0,
    "std_score": 10.0,
    "min_score": 60.0,
    "max_score": 95.0,
    "num_assessments": 8,
    "avg_submission_date": 90.0,
    "std_submission_date": 20.0,
    "score_range": 35.0,
    "total_clicks": 850,
    "avg_clicks": 100.0,
    "std_clicks": 30.0,
    "max_clicks": 200,
    "num_interactions": 15,
    "first_access": 10,
    "last_access": 200,
    "access_duration": 190,
    "avg_registration_date": -20.0,
    "num_unregistrations": 0
}

try:
    response = requests.post(f"{API_URL}/predict", json=normal_student)
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Full Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: At-Risk Student
print("\n[TEST 3] At-Risk Student Prediction...")
at_risk_student = {
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
    "avg_score": 35.0,
    "std_score": 12.0,
    "min_score": 15.0,
    "max_score": 55.0,
    "num_assessments": 3,
    "avg_submission_date": 180.0,
    "std_submission_date": 40.0,
    "score_range": 40.0,
    "total_clicks": 150,
    "avg_clicks": 20.0,
    "std_clicks": 15.0,
    "max_clicks": 80,
    "num_interactions": 5,
    "first_access": 100,
    "last_access": 200,
    "access_duration": 100,
    "avg_registration_date": -10.0,
    "num_unregistrations": 1
}

try:
    response = requests.post(f"{API_URL}/predict", json=at_risk_student)
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Full Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Minimal Data Test
print("\n[TEST 4] Minimal Required Data Test...")
minimal_data = {
    "code_module": "AAA",
    "code_presentation": "2013J",
    "gender": "M",
    "region": "East Anglian Region",
    "highest_education": "HE Qualification",
    "imd_band": "20-30%",
    "age_band": "35-55",
    "disability": "N",
    "studied_credits": 60,
    "num_of_prev_attempts": 0,
    "avg_score": 45.0,
    "std_score": 15.0,
    "min_score": 20.0,
    "max_score": 85.0,
    "num_assessments": 5,
    "avg_submission_date": 100.0,
    "std_submission_date": 30.0,
    "score_range": 65.0,
    "total_clicks": 500,
    "avg_clicks": 50.0,
    "std_clicks": 20.0,
    "max_clicks": 150,
    "num_interactions": 10,
    "first_access": 10,
    "last_access": 200,
    "access_duration": 190,
    "avg_registration_date": -15.0,
    "num_unregistrations": 0
}

try:
    response = requests.post(f"{API_URL}/predict", json=minimal_data)
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Full Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("✅ TESTS COMPLETED!")
print("="*60)