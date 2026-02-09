"""
API Testing Script for Student Anomaly Detection
Test all endpoints and verify responses
"""

import requests
import json
import time

# Configuration
API_URL = "http://localhost:5000"  # Change to your deployed URL
API_KEY = "your-api-key-here"  # If using authentication

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

CHEATER_PROFILE = {
    "code_module": "CCC",
    "code_presentation": "2014B",
    "gender": "M",
    "region": "London Region",
    "highest_education": "A Level or Equivalent",
    "imd_band": "40-50%",
    "age_band": "0-35",
    "disability": "N",
    "studied_credits": 60,
    "num_of_prev_attempts": 0,
    "avg_score": 92.0,  # Very high score
    "std_score": 5.0,   # Very consistent
    "min_score": 85,
    "max_score": 98,
    "num_assessments": 4,
    "avg_submission_date": 200,  # Last minute
    "std_submission_date": 10,
    "score_range": 13,
    "total_clicks": 120,  # Very low engagement
    "avg_clicks": 15,
    "std_clicks": 8,
    "max_clicks": 50,
    "num_interactions": 4,  # Minimal interaction
    "first_access": 180,  # Late start
    "last_access": 205,
    "access_duration": 25,  # Brief duration
    "avg_registration_date": -5,
    "num_unregistrations": 0
}


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_health_check():
    """Test the health check endpoint"""
    print_section("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Health check PASSED")
            return True
        else:
            print("✗ Health check FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_info():
    """Test the model info endpoint"""
    print_section("Testing Model Info Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/model_info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Model info PASSED")
            return True
        else:
            print("✗ Model info FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_prediction(student_data, label):
    """Test single prediction endpoint"""
    print_section(f"Testing Prediction: {label}")
    
    try:
        headers = {
            'Content-Type': 'application/json',
            # 'X-API-Key': API_KEY  # Uncomment if using authentication
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=student_data,
            headers=headers
        )
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.3f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Result:")
            print(f"  At Risk: {'YES' if result['isAnomaly'] else 'NO'}")
            print(f"  Risk Score: {result['riskScore']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Recommendation: {result['recommendation']}")
            
            if result.get('factors'):
                print(f"\n  Risk Factors:")
                for i, factor in enumerate(result['factors'], 1):
                    print(f"    {i}. {factor['name']} ({factor['impact']} impact) - Value: {factor['value']}")
            
            print(f"\n✓ Prediction test PASSED for {label}")
            return True
        else:
            print(f"✗ Prediction test FAILED for {label}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_section("Testing Batch Prediction")
    
    try:
        batch_data = {
            "students": [
                {**AT_RISK_STUDENT, "id_student": 1001},
                {**NORMAL_STUDENT, "id_student": 1002},
                {**CHEATER_PROFILE, "id_student": 1003}
            ]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=batch_data,
            headers=headers
        )
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.3f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nBatch Results:")
            print(f"  Total Predictions: {result['total']}")
            
            for pred in result['predictions']:
                status = "AT RISK" if pred['isAnomaly'] else "NORMAL"
                print(f"\n  Student {pred['student_id']}: {status}")
                print(f"    Risk Score: {pred['riskScore']:.3f}")
            
            print(f"\n✓ Batch prediction test PASSED")
            return True
        else:
            print(f"✗ Batch prediction test FAILED")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid data"""
    print_section("Testing Error Handling")
    
    try:
        # Test with missing data
        invalid_data = {"avg_score": 50}  # Missing most fields
        
        response = requests.post(
            f"{API_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [400, 500]:
            print("✓ Error handling test PASSED (correctly rejected invalid input)")
            return True
        else:
            print("⚠ Warning: API accepted invalid input")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "API TEST SUITE" + " "*29 + "║")
    print("╚" + "="*58 + "╝")
    print(f"\nTesting API at: {API_URL}")
    
    results = {
        'Health Check': test_health_check(),
        'Model Info': test_model_info(),
        'At-Risk Student': test_prediction(AT_RISK_STUDENT, "At-Risk Student"),
        'Normal Student': test_prediction(NORMAL_STUDENT, "Normal Student"),
        'Cheater Profile': test_prediction(CHEATER_PROFILE, "Potential Cheater"),
        'Batch Prediction': test_batch_prediction(),
        'Error Handling': test_error_handling()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for production.")
    else:
        print("⚠️  Some tests failed. Please check the API configuration.")


if __name__ == "__main__":
    run_all_tests()