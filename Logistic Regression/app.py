import streamlit as st
import pickle
import pandas as pd

# Load trained model, scaler, and imputer
model = pickle.load(open("log_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))

st.title("Titanic Survival Prediction - CSV Batch Test")
st.write("Upload a CSV file to predict survival for multiple passengers.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Preprocessing
    # Drop irrelevant columns if present
    for col in ["Name", "Ticket", "Cabin"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Encode categorical variables
    if "Sex" in df.columns:
        df["Sex_male"] = df["Sex"].apply(lambda x: 1 if x=="male" else 0)
        df = df.drop("Sex", axis=1)
    if "Embarked" in df.columns:
        df["Embarked_Q"] = df["Embarked"].apply(lambda x: 1 if x=="Q" else 0)
        df["Embarked_S"] = df["Embarked"].apply(lambda x: 1 if x=="S" else 0)
        df = df.drop("Embarked", axis=1)

    # Align columns with training data
    # Make sure all model features exist
    model_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.columns
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    # Impute missing values
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # Scale features
    df_scaled = scaler.transform(df_imputed)

    # Make predictions
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:,1]

    # Show results
    results = df.copy()
    results["Predicted_Survived"] = predictions
    results["Survival_Probability"] = probabilities
    st.write("Prediction Results:")
    st.dataframe(results)
