import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("ann_model.h5")

label_encoder_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
onehot_encoder_geography = pickle.load(open("onehot_encoder_geography.pkl", "rb"))
sc = pickle.load(open("scaler.pkl", "rb"))

st.title("Credit Card Customer Churn Prediction")

st.sidebar.header("User Input Parameters")

# Sidebar - Collects user input features into dataframe
credit_score = st.sidebar.slider("Credit Score", 350, 850, 500)
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
age = st.sidebar.slider("Age", 18, 90, 30)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.slider("Balance", 0, 250000, 50000)
num_of_products = st.sidebar.slider("Number of products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ("Yes", "No"))
is_active_member = st.sidebar.selectbox("Is Active Member?", ("Yes", "No"))
estimated_salary = st.sidebar.slider("Estimated Salary", 11, 100000, 50000)
geography = st.sidebar.selectbox(
    "Geography", onehot_encoder_geography.categories_[0]
)

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card == "Yes"],
    "IsActiveMember": [is_active_member == "Yes"],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
df_geo = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), df_geo], axis=1)

input_data_scaled = sc.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.subheader("Prediction Probability")
st.write(prediction_probability)

st.subheader("Customer Churn Prediction")
if prediction_probability > 0.5:
    st.write("Customer will not churn")
else:
    st.write("Customer will churn")
