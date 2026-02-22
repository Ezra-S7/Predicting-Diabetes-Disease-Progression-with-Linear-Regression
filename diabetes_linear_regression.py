"""
Predicting Diabetes Disease Progression with Linear Regression
==============================================================
This script applies linear regression to the Diabetes Dataset
(built into scikit-learn). The goal is to predict a quantitative
measure of disease progression one year after baseline, based on
ten physiological features such as BMI, blood pressure, and serum
measurements.

Dataset: 442 patients, 10 features, continuous target (disease progression score).

Author: Ezra Spivak
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load and inspect dataset

data = load_diabetes()
X, y = data.data, data.target

print("=" * 60)
print("Diabetes Disease Progression Dataset")
print("=" * 60)
print(f"  Patients  : {X.shape[0]}")
print(f"  Features  : {X.shape[1]}")
print(f"  Target    : quantitative disease progression score")
print(f"  Target range: {y.min():.1f} – {y.max():.1f}  (mean: {y.mean():.1f})")
print()

#Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print()

#Feature scaling - Standardizing features (mean=0, std=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Train the linear regression model

model = LinearRegression()
model.fit(X_train_scaled, y_train)

#Evaluate on the held-out test set

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 60)
print("Model Performance on Test Set")
print("=" * 60)
print(f"  R² Score                : {r2:.4f}")
print(f"  Root Mean Squared Error : {rmse:.2f}")
print()
print(f"  Interpretation: the model explains {r2*100:.1f}% of the variance")
print(f"  in disease progression scores, with a typical prediction")
print(f"  error of ±{rmse:.1f} points.")
print()

#5-fold cross-validation

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
print(f"  5-Fold CV R² : {cv_scores.mean():.4f}  (std: {cv_scores.std():.4f})")
print()

#Feature importance via regression coefficients

feature_importance = sorted(
    zip(data.feature_names, model.coef_),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("=" * 60)
print("Feature Coefficients (standardized)")
print("=" * 60)
print(f"  {'Feature':<10} {'Coefficient':>12}  {'Direction'}")
print("  " + "-" * 42)
for feature, coef in feature_importance:
    direction = "↑ higher progression" if coef > 0 else "↓ lower progression"
    print(f"  {feature:<10} {coef:>+12.4f}  {direction}")

print()
print("  Features with larger absolute coefficients have greater")
print("  influence on predicted disease progression.")
