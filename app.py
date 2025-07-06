# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load('model.pkl')
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

st.title("🌸 Iris Flower Classifier")

st.write("""
Enter the characteristics of the Iris flower to predict its species.
""")

# User input form
with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=features)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌼 Predicted Species: {species[prediction]}")

    # Probability bar chart
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=species)
    st.bar_chart(prob_df.T)

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
