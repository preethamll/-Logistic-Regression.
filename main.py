# main.py -- corrected preprocessing + logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, classification_report

# ---------- Load data ----------
df = pd.read_csv("data.csv")

print("Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------- Basic cleanup ----------
# Drop obvious useless columns
cols_to_drop = []
if "id" in df.columns:
    cols_to_drop.append("id")
# Drop unnamed trailing empty column(s)
for c in df.columns:
    if str(c).startswith("Unnamed"):
        cols_to_drop.append(c)
if cols_to_drop:
    print("Dropping columns:", cols_to_drop)
    df = df.drop(columns=cols_to_drop)

# ---------- Target encoding ----------
# If 'diagnosis' is the target, encode it: M -> 1, B -> 0
target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
print("Using target column:", target_col)

if df[target_col].dtype == object or not np.issubdtype(df[target_col].dtype, np.number):
    # common mapping for this dataset
    mapping = {"M": 1, "B": 0}
    if set(df[target_col].unique()) <= set(mapping.keys()):
        df[target_col] = df[target_col].map(mapping)
        print(f"Mapped target {mapping}")
    else:
        # fallback: label encoding
        df[target_col] = pd.Categorical(df[target_col]).codes
        print("Applied categorical codes to target")

# ---------- Identify feature columns ----------
X = df.drop(columns=[target_col])
y = df[target_col]

# Check for any remaining non-numeric columns in X
non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Non-numeric feature columns:", non_numeric)
if non_numeric:
    # Try to auto-encode simple categorical columns using one-hot
    print("One-hot encoding non-numeric features:", non_numeric)
    X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

# Convert all to numeric if possible
X = X.apply(pd.to_numeric, errors='coerce')

# ---------- Handle missing values ----------
missing = X.isna().sum().sum() + y.isna().sum()
print("Total missing (features + target):", missing)
if missing > 0:
    # Simple imputation: fill numeric NaNs with column mean; drop rows where target is NaN
    X = X.fillna(X.mean())
    # If target has NaNs, drop those rows
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    print("Filled numeric NaNs with column means and dropped rows with missing target.")

print("Final feature shape:", X.shape)
print("Final target shape:", y.shape)
print("Feature dtypes:\n", X.dtypes.value_counts())

# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ---------- Standardization ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # NOW all numeric
X_test = scaler.transform(X_test)

# ---------- Logistic Regression ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- Predictions & Metrics ----------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Results ---")
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC Score:", roc_auc)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ---------- ROC Curve ----------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], '--', linewidth=0.8)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("roc_curve.png")
print("Saved ROC curve as roc_curve.png")
plt.show()

# ---------- Sigmoid function (visual aid) ----------
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.figure()
plt.plot(z, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.savefig("sigmoid_curve.png")
print("Saved sigmoid curve as sigmoid_curve.png")
plt.show()
