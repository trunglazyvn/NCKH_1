import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, TEST_SIZE

def load_data():
    credits = pd.read_csv(DATA_PATH)
    x = credits.drop(columns=["Class"])
    y = credits["Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train_f, x_val, y_train_f, y_val = train_test_split(x_train, y_train, test_size=TEST_SIZE, random_state=42)

    scaler = StandardScaler()

    x_train_f[['Time', 'Amount']] = scaler.fit_transform(
        x_train_f[['Time', 'Amount']]
    )

    x_val[['Time', 'Amount']] = scaler.transform(
        x_val[['Time', 'Amount']]
    )       

    x_test[['Time', 'Amount']] = scaler.transform(
        x_test[['Time', 'Amount']]
    )
    
    return x_train_f, x_test, x_val, y_train_f, y_test, y_val, scaler