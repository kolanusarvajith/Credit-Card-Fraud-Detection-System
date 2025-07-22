import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_data
def train_model():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)

    X = data.drop('Class', axis=1)
    Y = data['Class']

    rfc = RandomForestClassifier(n_estimators=100, random_state=0)

    rfc.fit(X, Y)

    return rfc, data

model, data = train_model()

st.title("Credit Card Fraud Detection App 💳")
st.markdown("""
This application uses a trained **Random Forest** model to predict whether a credit card
transaction is fraudulent or legitimate based on its features.
Please enter the transaction details in the sidebar to get a prediction.
""")

st.sidebar.header("Transaction Features")
st.sidebar.markdown("Enter the details of the transaction below.")

def get_user_input():
    
    inputs = {}

    feature_columns = data.drop('Class', axis=1).columns

    for feature in feature_columns:
        min_val = -1e6
        max_val = 1e6
        mean_val = float(data[feature].mean())
        
        inputs[feature] = st.sidebar.number_input(
            label=f"Enter value for {feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return inputs

user_inputs = get_user_input()

if st.sidebar.button("Predict Transaction"):
    
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.error("This transaction is likely to be FRAUDULENT!", icon="🚨")
        st.info("It is highly recommended to block this transaction and contact the cardholder immediately.", icon="ℹ️")
        
    else:
        st.success("This transaction appears to be LEGITIMATE.", icon="✅")

    st.write("---")
    # st.subheader("Transaction Details Entered:")
    # st.json(user_inputs)