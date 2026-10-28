import joblib

from config import THRESHOLD
from dataloader import load_data

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

model = joblib.load("models/xgboost_model.pkl")

x_train, x_test, x_val, y_train, y_test, y_val, scaler = load_data()

y_prod = model.predict_proba(x_val)[:, 1]

y_pred = (y_prod >= THRESHOLD).astype(int)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, zero_division=0))
print("Recall:", recall_score(y_val, y_pred, zero_division=0))
print("F1:", f1_score(y_val, y_pred, zero_division=0))