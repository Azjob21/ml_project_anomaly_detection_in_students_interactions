# ... (previous imports)
import joblib
import warnings
import wandb
import os

warnings.filterwarnings('ignore')

# Initialize W&B
wandb.init(project="student-anomaly-detection", job_type="train")

print("="*60)
print("Starting Model Training...")
print("="*60)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/7] Loading datasets...")

# Update these paths to where you extracted the dataset
DATA_PATH = 'data/'  # Change this to your data folder path

# Log data as artifact (optional, good for versioning)
# artifact = wandb.Artifact('student-data', type='dataset')
# artifact.add_dir(DATA_PATH)
# wandb.log_artifact(artifact)

students = pd.read_csv(f'{DATA_PATH}studentInfo.csv')
assessments = pd.read_csv(f'{DATA_PATH}studentAssessment.csv')
vle = pd.read_csv(f'{DATA_PATH}studentVle.csv')
student_registration = pd.read_csv(f'{DATA_PATH}studentRegistration.csv')

print(f"✓ Students: {students.shape}")
print(f"✓ Assessments: {assessments.shape}")
print(f"✓ VLE: {vle.shape}")
print(f"✓ Registration: {student_registration.shape}")

# ... (Feature Engineering and Merge steps remain same, but for brevity I will focus on where I add W&B logging) ... 
# actually, I need to be careful with replace_file_content. I should probably use multi_replace or just read the file again to be safe about line numbers if I am not replacing everything. 
# BUT, since I want to wrap the whole thing or insert at specific points, maybe rewriting the file with the changes is safer given the complexity of insertions.
# Let's try to be precise with replace_file_content on the imports and the end.

# Wait, the tool definition says "This must be a complete drop-in replacement of the TargetContent". 
# I will use multi_replace to insert imports and then the end block.



# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/7] Loading datasets...")

# Update these paths to where you extracted the dataset
DATA_PATH = 'data/'  # Change this to your data folder path

students = pd.read_csv(f'{DATA_PATH}studentInfo.csv')
assessments = pd.read_csv(f'{DATA_PATH}studentAssessment.csv')
vle = pd.read_csv(f'{DATA_PATH}studentVle.csv')
student_registration = pd.read_csv(f'{DATA_PATH}studentRegistration.csv')

print(f"✓ Students: {students.shape}")
print(f"✓ Assessments: {assessments.shape}")
print(f"✓ VLE: {vle.shape}")
print(f"✓ Registration: {student_registration.shape}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[2/7] Creating features...")

# Assessment features
assessment_agg = assessments.groupby('id_student').agg({
    'score': ['mean', 'std', 'min', 'max', 'count'],
    'date_submitted': ['mean', 'std']
}).reset_index()

assessment_agg.columns = ['id_student', 'avg_score', 'std_score', 'min_score', 
                          'max_score', 'num_assessments', 'avg_submission_date', 
                          'std_submission_date']

assessment_agg['score_range'] = assessment_agg['max_score'] - assessment_agg['min_score']
assessment_agg['std_score'] = assessment_agg['std_score'].fillna(0)
assessment_agg['std_submission_date'] = assessment_agg['std_submission_date'].fillna(0)

# VLE features
vle_agg = vle.groupby('id_student').agg({
    'sum_click': ['sum', 'mean', 'std', 'max'],
    'date': ['count', 'min', 'max']
}).reset_index()

vle_agg.columns = ['id_student', 'total_clicks', 'avg_clicks', 'std_clicks', 
                   'max_clicks', 'num_interactions', 'first_access', 'last_access']

vle_agg['access_duration'] = vle_agg['last_access'] - vle_agg['first_access']
vle_agg['std_clicks'] = vle_agg['std_clicks'].fillna(0)

# Registration features
reg_features = student_registration.groupby('id_student').agg({
    'date_registration': 'mean',
    'date_unregistration': lambda x: x.notna().sum()
}).reset_index()

reg_features.columns = ['id_student', 'avg_registration_date', 'num_unregistrations']

print("✓ Features created")

# ============================================================================
# STEP 3: Merge and Prepare Data
# ============================================================================
print("\n[3/7] Merging datasets...")

df = students.copy()
df = df.merge(assessment_agg, on='id_student', how='left')
df = df.merge(vle_agg, on='id_student', how='left')
df = df.merge(reg_features, on='id_student', how='left')

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Create anomaly labels
df['is_anomaly'] = ((df['final_result'] == 'Fail') | 
                    (df['final_result'] == 'Withdrawn')).astype(int)

print(f"✓ Dataset merged: {df.shape}")
print(f"✓ Anomaly rate: {df['is_anomaly'].mean():.2%}")

# ============================================================================
# STEP 4: Encode Categorical Variables
# ============================================================================
print("\n[4/7] Encoding categorical variables...")

label_encoders = {}
categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                   'highest_education', 'imd_band', 'age_band', 'disability']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical features")

# ============================================================================
# STEP 5: Prepare Feature Matrix
# ============================================================================
print("\n[5/7] Preparing feature matrix...")

feature_cols = [col + '_encoded' for col in categorical_cols] + [
    'studied_credits', 'num_of_prev_attempts',
    'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
    'avg_submission_date', 'std_submission_date', 'score_range',
    'total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks',
    'num_interactions', 'first_access', 'last_access', 'access_duration',
    'avg_registration_date', 'num_unregistrations'
]

X = df[feature_cols].copy()
y = df['is_anomaly'].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ============================================================================
# STEP 6: Train Model
# ============================================================================
print("\n[6/7] Training Isolation Forest model...")

contamination_rate = min(float(y_train.mean()), 0.5)

model = IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination=contamination_rate,
    random_state=42,
    verbose=0
)

model.fit(X_train)
print("✓ Model trained successfully")

# Evaluate
# Evaluate
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

f1 = f1_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"  F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Log metrics to W&B
wandb.log({
    "f1_score": f1,
    "contamination": contamination_rate,
    "n_estimators": 200,
    "random_state": 42
})

# ============================================================================
# STEP 7: Save Models and Preprocessors
# ============================================================================
print("\n[7/7] Saving model files...")

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

# Save files
model_path = 'models/best_anomaly_model.pkl'
scaler_path = 'models/scaler.pkl'
encoders_path = 'models/label_encoders.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoders_path)

print(f"✓ Model saved to: {model_path}")
print(f"✓ Scaler saved to: {scaler_path}")
print(f"✓ Encoders saved to: {encoders_path}")

# Log artifacts to W&B
artifact = wandb.Artifact('anomaly-detection-model', type='model')
artifact.add_file(model_path)
artifact.add_file(scaler_path)
artifact.add_file(encoders_path)
wandb.log_artifact(artifact)

# Finish the run
wandb.finish()

print("\n" + "="*60)
print("✓ TRAINING COMPLETE! W&B Run finished.")
print("="*60)
print("\nYou can now run your Flask API with:")
print("  python app.py")
print("="*60)