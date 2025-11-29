#!/usr/bin/env python3
"""
====================================

This script walks through the entire machine learning pipeline from scratch:
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Export for FHE backend

Run this script to train a new model from scratch, replacing the pickle files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

import joblib
import os
import json
from datetime import datetime

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("COMPLETE CREDIT DEFAULT RISK PREDICTION ML PIPELINE")
print("="*80)
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# =============================================================================
# 1. LOAD THE DATASET
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

data_path = 'data/credit_card_default.csv'

if not os.path.exists(data_path):
    print(f" Dataset not found at {data_path}")
    print(" Please ensure the dataset is in the data/ directory")
    exit(1)

df = pd.read_csv(data_path)
print(f" Dataset loaded successfully!")
print(f" Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f" Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n First 5 rows:")
print(df.head())

print("\n Dataset Info:")
print(df.info())


# =============================================================================
# 2. RENAME COLUMNS FOR CLARITY
# =============================================================================
print("\n" + "="*80)
print("STEP 2: RENAMING COLUMNS")
print("="*80)

feature_names = {
    'X1': 'LIMIT_BAL',
    'X2': 'SEX',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',
    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',
    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6',
    'Y': 'DEFAULT'
}

df = df.rename(columns=feature_names)
print(" Columns renamed for clarity")


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*80)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Check for missing values
print("\n Checking for missing values...")
missing = df.isnull().sum()
if missing.sum() == 0:
    print(" No missing values found!")
else:
    print("  Missing values found:")
    print(missing[missing > 0])

# Statistical summary
print("\n Statistical Summary:")
print(df.describe())

# Target variable distribution
print("\n Target Variable Distribution:")
target_counts = df['DEFAULT'].value_counts()
target_pct = df['DEFAULT'].value_counts(normalize=True) * 100

print(f"No Default (0): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"Default (1): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"⚖️ Class Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")

# Demographic analysis
print("\n Demographic Analysis:")
print(f"Gender: {df['SEX'].value_counts().to_dict()} (1=Male, 2=Female)")
print(f"Education: {df['EDUCATION'].value_counts().to_dict()} (1=Grad, 2=Uni, 3=HS, 4=Other)")
print(f"Marriage: {df['MARRIAGE'].value_counts().to_dict()} (1=Married, 2=Single, 3=Other)")
print(f"Age: Mean={df['AGE'].mean():.1f}, Std={df['AGE'].std():.1f}, Range=[{df['AGE'].min()}, {df['AGE'].max()}]")

# Correlation with target
print("\n Top 10 Features Correlated with DEFAULT:")
correlations = df.corr()['DEFAULT'].sort_values(ascending=False)
print(correlations[1:11])  # Exclude DEFAULT itself


# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DATA CLEANING")
print("="*80)

# Check for duplicates
print("\n Checking for duplicates...")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f" Removed {duplicates} duplicate rows")
else:
    print(" No duplicates found")

# Clean categorical features
print("\n Cleaning categorical features...")

# Education: Map invalid values (0, 5, 6) to 'Others' (4)
invalid_edu = df['EDUCATION'].isin([0, 5, 6]).sum()
if invalid_edu > 0:
    print(f"  Found {invalid_edu} rows with invalid EDUCATION values")
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    print(" Mapped invalid EDUCATION values to 'Others' (4)")

# Marriage: Map invalid value (0) to 'Others' (3)
invalid_mar = df['MARRIAGE'].isin([0]).sum()
if invalid_mar > 0:
    print(f"  Found {invalid_mar} rows with invalid MARRIAGE value")
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    print(" Mapped invalid MARRIAGE values to 'Others' (3)")

print("\n Data cleaning complete!")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING")
print("="*80)

print("\n Creating new features...")

# 1. Average payment delay
df['AVG_PAY_DELAY'] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)

# 2. Total bill amount
df['TOTAL_BILL'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)

# 3. Total payment amount
df['TOTAL_PAYMENT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)

# 4. Payment to bill ratio
df['PAYMENT_RATIO'] = np.where(df['TOTAL_BILL'] > 0, 
                                df['TOTAL_PAYMENT'] / df['TOTAL_BILL'], 
                                0)

# 5. Credit utilization
avg_bill = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
df['CREDIT_UTILIZATION'] = np.where(df['LIMIT_BAL'] > 0,
                                     avg_bill / df['LIMIT_BAL'],
                                     0)

# 6. Number of times payment delayed
df['NUM_DELAYS'] = (df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1)

print(" Created 6 new features:")
print("   - AVG_PAY_DELAY: Average payment delay across 6 months")
print("   - TOTAL_BILL: Sum of all bill amounts")
print("   - TOTAL_PAYMENT: Sum of all payment amounts")
print("   - PAYMENT_RATIO: Total payment / Total bill")
print("   - CREDIT_UTILIZATION: Average bill / Credit limit")
print("   - NUM_DELAYS: Number of months with payment delay")

print(f"\n New dataset shape: {df.shape}")


# =============================================================================
# 6. PREPARE DATA FOR MODELING
# =============================================================================
print("\n" + "="*80)
print("STEP 6: PREPARING DATA FOR MODELING")
print("="*80)

# Use 20 features (excluding demographic features to prevent bias)
# Removed: SEX, EDUCATION, MARRIAGE (to ensure fairness and prevent discrimination)
original_features = ['LIMIT_BAL',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

X = df[original_features].copy()
y = df['DEFAULT'].copy()

print(f" Using {len(original_features)} features (EXCLUDING demographic features for fairness)")
print(f" Removed: SEX, EDUCATION, MARRIAGE to prevent discrimination")
print(f" Features shape: {X.shape}")
print(f" Target shape: {y.shape}")
print(f" Target distribution: {y.value_counts().to_dict()}")

# Split data
print("\n Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f" Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f" Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f" Train default rate: {y_train.mean():.2%}")
print(f" Test default rate: {y_test.mean():.2%}")

# Feature Scaling
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Features scaled using StandardScaler")


# =============================================================================
# 7. MODEL TRAINING
# =============================================================================
print("\n" + "="*80)
print("STEP 7: MODEL TRAINING AND COMPARISON")
print("="*80)

models = {
    'Logistic Regression': CalibratedClassifierCV(
        LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
        method='sigmoid',  # Platt Scaling
        cv=5  # 5-fold cross-validation for calibration
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=3,  # Very shallow trees to match LR predictions
        min_samples_leaf=50,  # High threshold for more linear-like behavior
        random_state=42, 
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,  # Fewer trees for conservative predictions
        max_depth=2,  # Very shallow trees to match LR
        learning_rate=0.05,  # Slow learning for closer agreement with LR
        min_samples_leaf=40,  # High threshold to prevent overfitting
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training {name}...")
    print(f"{'='*80}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n {name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    print(f"\n Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                No     Yes")
    print(f"Actual No    {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"       Yes   {cm[1][0]:6d} {cm[1][1]:6d}")


# =============================================================================
# 8. MODEL COMPARISON
# =============================================================================
print("\n" + "="*80)
print("STEP 8: MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'F1 Score': [results[m]['f1_score'] for m in results],
    'ROC AUC': [results[m]['roc_auc'] for m in results]
})

comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
print("\n Model Comparison Summary:")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n Best Model (by ROC AUC): {best_model_name}")


# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*80)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

lr_model = results['Logistic Regression']['model']
# Access base estimator coefficients from calibrated model
if hasattr(lr_model, 'calibrated_classifiers_'):
    # CalibratedClassifierCV - access the base estimator
    base_estimator = lr_model.calibrated_classifiers_[0].estimator
    coefficients = base_estimator.coef_[0]
else:
    # Regular model
    coefficients = lr_model.coef_[0]

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n Top 15 Most Important Features (Logistic Regression):")
print(feature_importance.head(15).to_string(index=False))


# =============================================================================
# 10. SAVE ALL MODELS
# =============================================================================
print("\n" + "="*80)
print("STEP 10: SAVING ALL MODELS")
print("="*80)

output_dir = 'backend/models'
os.makedirs(output_dir, exist_ok=True)

# Save scaler (shared by all models)
scaler_path = os.path.join(output_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f" Scaler saved to: {scaler_path}")

# Save feature names (shared by all models)
feature_names_path = os.path.join(output_dir, 'feature_names.txt')
with open(feature_names_path, 'w') as f:
    for feature in X_train.columns:
        f.write(f"{feature}\n")
print(f" Feature names saved to: {feature_names_path}")

# Save all models with their metadata
all_models_metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': len(X_train.columns),
    'feature_names': list(X_train.columns),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'models': {}
}

print("\n Saving all trained models...")
for name, result in results.items():
    # Create safe filename
    safe_name = name.lower().replace(' ', '_')
    model_path = os.path.join(output_dir, f'{safe_name}.pkl')
    
    # Save model
    joblib.dump(result['model'], model_path)
    print(f" {name} saved to: {model_path}")
    
    # Get model parameters
    model_params = result['model'].get_params()
    
    # Store metadata
    all_models_metadata['models'][name] = {
        'filename': f'{safe_name}.pkl',
        'accuracy': float(result['accuracy']),
        'f1_score': float(result['f1_score']),
        'roc_auc': float(result['roc_auc']),
        'fhe_compatible': (name == 'Logistic Regression'),
        'model_type': type(result['model']).__name__,
        'parameters': {
            'n_estimators': model_params.get('n_estimators', 'N/A'),
            'max_iter': model_params.get('max_iter', 'N/A'),
            'random_state': model_params.get('random_state', 'N/A'),
            'solver': model_params.get('solver', 'N/A'),
            'n_jobs': model_params.get('n_jobs', 'N/A')
        }
    }

# Save combined metadata
metadata_path = os.path.join(output_dir, 'all_models_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(all_models_metadata, f, indent=4)
print(f" All models metadata saved to: {metadata_path}")

# Also save legacy metadata for backward compatibility
legacy_metadata = {
    'model_name': 'Logistic Regression',
    'n_features': len(X_train.columns),
    'feature_names': list(X_train.columns),
    'accuracy': float(results['Logistic Regression']['accuracy']),
    'f1_score': float(results['Logistic Regression']['f1_score']),
    'roc_auc': float(results['Logistic Regression']['roc_auc']),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
legacy_path = os.path.join(output_dir, 'model_metadata.json')
with open(legacy_path, 'w') as f:
    json.dump(legacy_metadata, f, indent=4)

# Create copy for default model (Logistic Regression - FHE compatible)
import shutil
default_model_path = os.path.join(output_dir, 'credit_model.pkl')
shutil.copy(os.path.join(output_dir, 'logistic_regression.pkl'), default_model_path)
print(f" Default model (Logistic Regression) copied to: {default_model_path}")

print("\n All models saved! Default: Logistic Regression (FHE-compatible)")


# =============================================================================
# 11. TEST SAVED MODEL
# =============================================================================
print("\n" + "="*80)
print("STEP 11: TESTING SAVED MODEL")
print("="*80)

# Reload model and scaler
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

print(" Model and scaler loaded successfully")

# Test with samples
n_samples = 5
test_samples = X_test.iloc[:n_samples]
test_labels = y_test.iloc[:n_samples]

print(f"\n Testing with {n_samples} samples...\n")

for i in range(n_samples):
    sample = test_samples.iloc[i:i+1]
    true_label = test_labels.iloc[i]
    
    # Scale and predict
    sample_scaled = loaded_scaler.transform(sample)
    prediction = loaded_model.predict(sample_scaled)[0]
    probability = loaded_model.predict_proba(sample_scaled)[0][1]
    
    # Display results
    status = "✅" if prediction == true_label else "❌"
    print(f"Sample {i+1}: {status}")
    print(f"  True Label: {'DEFAULT' if true_label == 1 else 'NO DEFAULT'}")
    print(f"  Prediction: {'DEFAULT' if prediction == 1 else 'NO DEFAULT'}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Confidence: {'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'}")
    print()


# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print(" COMPLETE ML PIPELINE SUMMARY")
print("="*80)

print(f"\n Dataset:")
print(f"   - Total samples: {len(df):,}")
print(f"   - Features: {len(X_train.columns)}")
print(f"   - Default rate: {df['DEFAULT'].mean():.2%}")

print(f"\n Model Performance:")
print(f"   - Model: Logistic Regression (default)")
print(f"   - Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
print(f"   - F1 Score: {results['Logistic Regression']['f1_score']:.4f}")
print(f"   - ROC AUC: {results['Logistic Regression']['roc_auc']:.4f}")

print(f"\n Saved Files:")
print(f"   - {model_path}")
print(f"   - {scaler_path}")
print(f"   - {feature_names_path}")
print(f"   - {metadata_path}")

print(f"\n Next Steps:")
print(f"   1. ✅ Model is ready for FHE backend")
print(f"   2. Start backend: cd backend && uvicorn main:app --reload")
print(f"   3. Start frontend: cd frontend && streamlit run streamlit_app.py")
print(f"   4. Test API: http://localhost:8000/docs")
print(f"   5. Use web interface: http://localhost:8501")

print("\n" + "="*80)
print(" Model trained and saved successfully!")
print("="*80)

