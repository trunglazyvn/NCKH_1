from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def build_model():
    
    model = XGBClassifier(random_state=42)
    
    return model