import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load Model from Hugging Face which was uploaded in test.py step.
MODEL_REPO = "hsaluja431/tourism-model"
MODEL_FILENAME = "best_tourism_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
model = joblib.load(model_path)

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.45

# Streamlit UI
st.title("Wellness Tourism Package Prediction App")
st.write("Fill in the customer details to predict whether they will purchase the travel package.")


# Input Fields
# Numerical Inputs
Age = st.number_input("Age", min_value=18, max_value=90, value=30)
DurationOfPitch = st.number_input("Duration of Sales Pitch (minutes)", min_value=0, max_value=100, value=20)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1500000, value=50000)

# Categorical Inputs
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "AVP","VP"])

# Convert Inputs to DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation
}])

# Prediction Button
if st.button("Predict Purchase Likelihood"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob >= CLASSIFICATION_THRESHOLD else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Customer is **LIKELY** to purchase the package (Probability: {prob:.2f})")
    else:
        st.error(f"Customer is **NOT likely** to purchase the package (Probability: {prob:.2f})")

