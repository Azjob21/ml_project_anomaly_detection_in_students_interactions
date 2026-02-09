import React, { useState } from "react";
import {
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Users,
  Activity,
  Upload,
} from "lucide-react";

const StudentAnomalyDetector = () => {
  const [formData, setFormData] = useState({
    code_module: "AAA",
    code_presentation: "2013J",
    gender: "M",
    region: "East Anglian Region",
    highest_education: "HE Qualification",
    imd_band: "20-30%",
    age_band: "35-55",
    disability: "N",
    studied_credits: 60,
    num_of_prev_attempts: 0,
    avg_score: 50,
    std_score: 15,
    min_score: 20,
    max_score: 85,
    num_assessments: 5,
    avg_submission_date: 100,
    std_submission_date: 30,
    score_range: 65,
    total_clicks: 500,
    avg_clicks: 50,
    std_clicks: 20,
    max_clicks: 150,
    num_interactions: 10,
    first_access: 10,
    last_access: 200,
    access_duration: 190,
    avg_registration_date: -15,
    num_unregistrations: 0,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("basic");

  // Call the real API
  const predictAnomaly = async () => {
    setLoading(true);

    try {
      // Change this URL to your deployed API URL
      const API_URL = "http://localhost:5000";

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      setPrediction({
        isAnomaly: data.isAnomaly,
        riskScore: data.riskScore,
        confidence: data.confidence,
        factors: data.factors || [],
      });
    } catch (error) {
      console.error("Prediction error:", error);
      alert(
        "Failed to get prediction. Make sure the API is running on http://localhost:5000"
      );

      // Fallback to mock prediction if API fails
      const riskScore = calculateRiskScore(formData);
      const isAnomaly = riskScore > 0.5;
      setPrediction({
        isAnomaly,
        riskScore,
        confidence: 0.85,
        factors: getTopFactors(formData, isAnomaly),
      });
    } finally {
      setLoading(false);
    }
  };

  const calculateRiskScore = (data) => {
    let score = 0;

    // Low assessment scores
    if (data.avg_score < 40) score += 0.3;
    else if (data.avg_score < 50) score += 0.15;

    // Low engagement
    if (data.total_clicks < 300) score += 0.2;
    if (data.num_interactions < 8) score += 0.15;

    // Few assessments
    if (data.num_assessments < 4) score += 0.15;

    // Previous attempts
    if (data.num_of_prev_attempts > 0) score += 0.1 * data.num_of_prev_attempts;

    // Late submissions
    if (data.avg_submission_date > 150) score += 0.15;

    return Math.min(score, 1);
  };

  const getTopFactors = (data, isAnomaly) => {
    const factors = [];

    if (data.avg_score < 50) {
      factors.push({
        name: "Low Average Score",
        impact: "high",
        value: data.avg_score,
      });
    }
    if (data.total_clicks < 400) {
      factors.push({
        name: "Low Engagement",
        impact: "high",
        value: data.total_clicks,
      });
    }
    if (data.num_assessments < 5) {
      factors.push({
        name: "Few Assessments",
        impact: "medium",
        value: data.num_assessments,
      });
    }
    if (data.num_of_prev_attempts > 0) {
      factors.push({
        name: "Previous Attempts",
        impact: "medium",
        value: data.num_of_prev_attempts,
      });
    }

    return factors.slice(0, 4);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value) || 0,
    }));
  };

  const loadSampleData = (type) => {
    if (type === "risk") {
      setFormData({
        ...formData,
        avg_score: 35,
        total_clicks: 200,
        num_interactions: 5,
        num_assessments: 3,
        avg_submission_date: 180,
        num_of_prev_attempts: 2,
      });
    } else {
      setFormData({
        ...formData,
        avg_score: 75,
        total_clicks: 800,
        num_interactions: 15,
        num_assessments: 8,
        avg_submission_date: 90,
        num_of_prev_attempts: 0,
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-800 mb-2">
                Student At-Risk Detection System
              </h1>
              <p className="text-gray-600">
                AI-powered early warning system for online education
              </p>
            </div>
            <Activity className="w-16 h-16 text-blue-600" />
          </div>

          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="bg-blue-50 rounded-lg p-4">
              <Users className="w-8 h-8 text-blue-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">28,785</div>
              <div className="text-sm text-gray-600">Students Analyzed</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <TrendingUp className="w-8 h-8 text-green-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">89.4%</div>
              <div className="text-sm text-gray-600">Model Accuracy</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <CheckCircle className="w-8 h-8 text-purple-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">
                Isolation Forest
              </div>
              <div className="text-sm text-gray-600">Best Model</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Form */}
          <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">
                Student Information
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={() => loadSampleData("risk")}
                  className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200"
                >
                  Load At-Risk Sample
                </button>
                <button
                  onClick={() => loadSampleData("normal")}
                  className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200"
                >
                  Load Normal Sample
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-6 border-b">
              {["basic", "academic", "engagement"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 font-medium capitalize transition-colors ${
                    activeTab === tab
                      ? "text-blue-600 border-b-2 border-blue-600"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Basic Info Tab */}
            {activeTab === "basic" && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Module Code
                  </label>
                  <select
                    name="code_module"
                    value={formData.code_module}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="AAA">AAA</option>
                    <option value="BBB">BBB</option>
                    <option value="CCC">CCC</option>
                    <option value="DDD">DDD</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Gender
                  </label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Education Level
                  </label>
                  <select
                    name="highest_education"
                    value={formData.highest_education}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Lower Than A Level">
                      Lower Than A Level
                    </option>
                    <option value="A Level or Equivalent">
                      A Level or Equivalent
                    </option>
                    <option value="HE Qualification">HE Qualification</option>
                    <option value="Post Graduate Qualification">
                      Post Graduate
                    </option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Age Band
                  </label>
                  <select
                    name="age_band"
                    value={formData.age_band}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="0-35">0-35</option>
                    <option value="35-55">35-55</option>
                    <option value="55<=">55+</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Disability
                  </label>
                  <select
                    name="disability"
                    value={formData.disability}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="N">No</option>
                    <option value="Y">Yes</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Studied Credits
                  </label>
                  <input
                    type="number"
                    name="studied_credits"
                    value={formData.studied_credits}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            )}

            {/* Academic Performance Tab */}
            {activeTab === "academic" && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Average Score (0-100)
                  </label>
                  <input
                    type="number"
                    name="avg_score"
                    value={formData.avg_score}
                    onChange={handleInputChange}
                    min="0"
                    max="100"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Assessments
                  </label>
                  <input
                    type="number"
                    name="num_assessments"
                    value={formData.num_assessments}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Minimum Score
                  </label>
                  <input
                    type="number"
                    name="min_score"
                    value={formData.min_score}
                    onChange={handleInputChange}
                    min="0"
                    max="100"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Maximum Score
                  </label>
                  <input
                    type="number"
                    name="max_score"
                    value={formData.max_score}
                    onChange={handleInputChange}
                    min="0"
                    max="100"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Previous Attempts
                  </label>
                  <input
                    type="number"
                    name="num_of_prev_attempts"
                    value={formData.num_of_prev_attempts}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Avg Submission Date
                  </label>
                  <input
                    type="number"
                    name="avg_submission_date"
                    value={formData.avg_submission_date}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            )}

            {/* Engagement Tab */}
            {activeTab === "engagement" && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Total Clicks
                  </label>
                  <input
                    type="number"
                    name="total_clicks"
                    value={formData.total_clicks}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Average Clicks
                  </label>
                  <input
                    type="number"
                    name="avg_clicks"
                    value={formData.avg_clicks}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Interactions
                  </label>
                  <input
                    type="number"
                    name="num_interactions"
                    value={formData.num_interactions}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Maximum Clicks
                  </label>
                  <input
                    type="number"
                    name="max_clicks"
                    value={formData.max_clicks}
                    onChange={handleInputChange}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    First Access Day
                  </label>
                  <input
                    type="number"
                    name="first_access"
                    value={formData.first_access}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Last Access Day
                  </label>
                  <input
                    type="number"
                    name="last_access"
                    value={formData.last_access}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            )}

            <button
              onClick={predictAnomaly}
              disabled={loading}
              className="w-full mt-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Predict Risk Level"}
            </button>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {prediction && (
              <>
                <div
                  className={`bg-white rounded-2xl shadow-lg p-6 ${
                    prediction.isAnomaly
                      ? "ring-2 ring-red-500"
                      : "ring-2 ring-green-500"
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-gray-800">
                      Prediction
                    </h3>
                    {prediction.isAnomaly ? (
                      <AlertCircle className="w-8 h-8 text-red-500" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    )}
                  </div>

                  <div
                    className={`text-center py-6 rounded-xl ${
                      prediction.isAnomaly ? "bg-red-50" : "bg-green-50"
                    }`}
                  >
                    <div
                      className={`text-4xl font-bold mb-2 ${
                        prediction.isAnomaly ? "text-red-600" : "text-green-600"
                      }`}
                    >
                      {prediction.isAnomaly ? "AT RISK" : "ON TRACK"}
                    </div>
                    <div className="text-gray-600">
                      Risk Score: {(prediction.riskScore * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                      Risk Level
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full transition-all ${
                          prediction.riskScore > 0.7
                            ? "bg-red-500"
                            : prediction.riskScore > 0.4
                            ? "bg-yellow-500"
                            : "bg-green-500"
                        }`}
                        style={{ width: `${prediction.riskScore * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    Key Factors
                  </h3>
                  <div className="space-y-3">
                    {prediction.factors.map((factor, idx) => (
                      <div
                        key={idx}
                        className="border-l-4 border-blue-500 pl-3"
                      >
                        <div className="flex items-center justify-between">
                          <div className="font-medium text-gray-800">
                            {factor.name}
                          </div>
                          <span
                            className={`text-xs px-2 py-1 rounded ${
                              factor.impact === "high"
                                ? "bg-red-100 text-red-700"
                                : factor.impact === "medium"
                                ? "bg-yellow-100 text-yellow-700"
                                : "bg-blue-100 text-blue-700"
                            }`}
                          >
                            {factor.impact}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600">
                          Value: {factor.value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {prediction.isAnomaly && (
                  <div className="bg-orange-50 border border-orange-200 rounded-2xl p-6">
                    <h3 className="text-lg font-bold text-orange-800 mb-3">
                      Recommendations
                    </h3>
                    <ul className="space-y-2 text-sm text-orange-900">
                      <li className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>
                          Schedule one-on-one meeting with academic advisor
                        </span>
                      </li>
                      <li className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>
                          Provide additional learning resources and support
                        </span>
                      </li>
                      <li className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>Monitor engagement metrics weekly</span>
                      </li>
                      <li className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>Consider peer mentoring or study groups</span>
                      </li>
                    </ul>
                  </div>
                )}
              </>
            )}

            {!prediction && (
              <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
                <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">
                  Enter student information and click "Predict Risk Level" to
                  analyze
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 bg-white rounded-2xl shadow-lg p-6">
          <div className="text-center text-sm text-gray-600">
            <p className="font-semibold mb-2">Model Information</p>
            <p>
              Algorithm: Isolation Forest (Tuned) | Accuracy: 89.4% | F1-Score:
              0.8741
            </p>
            <p className="mt-2 text-xs">
              Dataset: Open University Learning Analytics (28,785 students)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentAnomalyDetector;
