# test_model.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from stats_tests import COLUMNS_OF_INTEREST, TARGET, get_validation_data, print_section

def load_model_components():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    model = joblib.load('random_forest_model.pkl')
    return scaler, pca, model

def predict_new_data(new_data, scaler, pca, model):
    new_data = new_data[COLUMNS_OF_INTEREST]
    X_scaled = scaler.transform(new_data)
    X_pca = pca.transform(X_scaled)
    predictions = model.predict(X_pca)
    return predictions

def evaluate_model(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return accuracy, report

def main():
    scaler, pca, model = load_model_components()
    df_val = get_validation_data()
    true_labels = df_val[TARGET]
    predictions = predict_new_data(df_val, scaler, pca, model)
    accuracy, report = evaluate_model(true_labels, predictions)
    
    print_section("Final Results:")  
    print("->Chosen Model: Random Forests Model")
    print("Model Accuracy With The Validation Data:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()