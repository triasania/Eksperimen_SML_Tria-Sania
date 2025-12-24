import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def run_preprocessing():
    print("Mulai Preprocessing...")
    
    # 1. Load Data
    df = pd.read_csv('dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # 2. Cleaning & Imputasi (Sesuai Notebook Cell 3)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    # 3. Label Encoding
    binary_map = {"Yes": 1, "No": 0}
    for col in ["Partner","Dependents","PhoneService","PaperlessBilling","Churn"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(binary_map)

    # 4. One-Hot Encoding
    ohe_cols = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                "Contract","PaymentMethod"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # 5. Drop Columns
    if "customerID" in df.columns: df.drop(columns=["customerID"], inplace=True)
    if "gender" in df.columns: df.drop(columns=["gender"], inplace=True)

    # 6. Split Data
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Scaling
    num_cols = ["tenure","MonthlyCharges","TotalCharges"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    
    # Simpan Scaler
    os.makedirs('preprocessing/dataset_preprocessing', exist_ok=True)
    joblib.dump(scaler, 'preprocessing/dataset_preprocessing/scaler.pkl')

    # 8. SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 9. Simpan CSV Bersih
    train_data = pd.concat([X_train_res, y_train_res], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('preprocessing/dataset_preprocessing/train_processed.csv', index=False)
    test_data.to_csv('preprocessing/dataset_preprocessing/test_processed.csv', index=False)
    
    print("Preprocessing Selesai! File tersimpan.")

if __name__ == "__main__":
    run_preprocessing()