"""
Five Classification Algorithms + Hybrid Ensembles
Dataset: Air Quality (UCI id=360)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("Loading Air Quality dataset (UCI id=360)")
print("=" * 60)

air_quality = fetch_ucirepo(id=360)
X_raw = air_quality.data.features
y_raw = air_quality.data.targets

print(f"Original X shape: {X_raw.shape}")
print(f"X columns: {list(X_raw.columns)}")

# Check if y is None
if y_raw is None or (hasattr(y_raw, 'shape') and y_raw.shape[1] == 0):
    print("\nTarget column is not provided separately.")
    print("Using 'CO(GT)' column from features as target.")
    print("This is the standard target for Air Quality dataset.")
    
    target_col = 'CO(GT)'
    
    if target_col not in X_raw.columns:
        # Try to find it case-insensitively
        for c in X_raw.columns:
            if 'co' in c.lower() and 'gt' in c.lower():
                target_col = c
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in features")
    
    y_series = pd.to_numeric(X_raw[target_col], errors='coerce')
    X_raw = X_raw.drop(columns=[target_col])
    print(f"Using target column: {target_col}")
else:
    y_series = y_raw.iloc[:, 0] if isinstance(y_raw, pd.DataFrame) else pd.Series(y_raw)
    y_series = pd.to_numeric(y_series, errors='coerce')

print(f"X shape after target extraction: {X_raw.shape}")
print(f"y length: {len(y_series)}")

# Remove rows with NaN target
valid_mask = y_series.notna().values
X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
y_series = y_series.loc[valid_mask].reset_index(drop=True)

# Drop Date/Time columns
drop_cols = [c for c in X_raw.columns if c.strip().lower() in ['date', 'time']]
if drop_cols:
    print(f"Dropping columns: {drop_cols}")
    X_raw = X_raw.drop(columns=drop_cols)

# Keep only numeric columns
X_raw = X_raw.select_dtypes(include=[np.number])
print(f"Features after numeric filter: {X_raw.shape}")

# Convert target to binary (above/below median)
threshold = y_series.median()
y_binary = (y_series > threshold).astype(int).values
print(f"Threshold (median CO(GT)): {threshold:.4f}")
print(f"Class distribution: {np.bincount(y_binary)}")


# ============================================================
# 2. Preprocessing
# ============================================================
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

non_const = np.std(X_scaled, axis=0) > 1e-10
X_scaled = X_scaled[:, non_const]
print(f"After removing constant features: {X_scaled.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================
# 3. Five Base Classifiers
# ============================================================
base_classifiers = {
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB()
}


# ============================================================
# 4. Run Each Classifier Separately
# ============================================================
print("\n" + "=" * 60)
print("Training base classifiers")
print("=" * 60)

results = {}

for name, clf in base_classifiers.items():
    print(f"\n{name}...")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    results[name] = {'accuracy': acc, 'f1': f1}
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")


# ============================================================
# 5. Hybrid Ensembles
# ============================================================
print("\n" + "=" * 60)
print("Training Hybrid Ensembles")
print("=" * 60)

# --- Soft Voting ---
voting_clf = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
voting_f1 = f1_score(y_test, voting_pred, zero_division=0)
print(f"\nSoft Voting:    Acc={voting_acc:.4f} | F1={voting_f1:.4f}")

# --- Stacking ---
stacking_clf = StackingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)
stacking_f1 = f1_score(y_test, stacking_pred, zero_division=0)
print(f"Stacking:       Acc={stacking_acc:.4f} | F1={stacking_f1:.4f}")

# --- Weighted Voting ---
voting_weighted = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='soft',
    weights=[2, 1, 1, 2, 1]
)
voting_weighted.fit(X_train, y_train)
weighted_pred = voting_weighted.predict(X_test)
weighted_acc = accuracy_score(y_test, weighted_pred)
weighted_f1 = f1_score(y_test, weighted_pred, zero_division=0)
print(f"Weighted Voting: Acc={weighted_acc:.4f} | F1={weighted_f1:.4f}")


# ============================================================
# 6. Final Comparison
# ============================================================
print("\n" + "=" * 60)
print("Final Comparison")
print("=" * 60)

comparison = pd.DataFrame({
    'Algorithm': list(results.keys()) + [
        'Soft Voting (Hybrid)',
        'Stacking (Hybrid)',
        'Weighted Voting (Hybrid)'
    ],
    'Accuracy': [results[k]['accuracy'] for k in results] + [
        voting_acc, stacking_acc, weighted_acc
    ],
    'F1-Score': [results[k]['f1'] for k in results] + [
        voting_f1, stacking_f1, weighted_f1
    ]
})

comparison = comparison.sort_values('Accuracy', ascending=False).reset_index(drop=True)
comparison.index = comparison.index + 1

print("\n" + comparison.to_string())

best = comparison.iloc[0]
print(f"\nBest Model: {best['Algorithm']}")
print(f"  Accuracy: {best['Accuracy']:.4f}")
print(f"  F1-Score: {best['F1-Score']:.4f}")

comparison.to_csv('classification_results.csv', index=False)
print("\nResults saved to 'classification_results.csv'")
print("Done!")