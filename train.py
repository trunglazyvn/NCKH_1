from dataloader import load_data
from model import build_model
import os
import joblib

def train():
    
    x_train, x_test, x_val, y_train, y_test, y_val, scaler = load_data()
    
    model = build_model()
    
    model.fit(x_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/xgboost_model.pkl")
    
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Huấn luyện hoàn thành")
    
if __name__ == "__main__":
    train()