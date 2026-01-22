# smart_loan_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------- App Title -----------------
st.title("Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

# ----------------- Load & preprocess dataset -----------------
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/kathiyashashwinireddy/Downloads/train_loan.csv')
    
    # Fill missing values
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(1, inplace=True)
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents',
                                             'Education', 'Self_Employed', 'Property_Area'], drop_first=True)
    return df_encoded

df_encoded = load_data()

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Enter Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=100)
credit_history = st.sidebar.selectbox("Credit History", options=["Yes", "No"])
employment_status = st.sidebar.selectbox("Employment Status", options=["Employed", "Self_Employed"])
property_area = st.sidebar.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# ----------------- Model Selection -----------------
st.sidebar.header("Select SVM Kernel")
kernel_choice = st.sidebar.radio("SVM Kernel", options=["Linear SVM", "Polynomial SVM", "RBF SVM"])

# ----------------- Prepare input for prediction -----------------
def prepare_input():
    # Create a dataframe with all columns
    input_df = pd.DataFrame(columns=df_encoded.drop(['Loan_ID','Loan_Status'],axis=1).columns)
    
    # Fill numeric
    input_df.loc[0, 'ApplicantIncome'] = app_income
    input_df.loc[0, 'CoapplicantIncome'] = 0  # assume no co-applicant for simplicity
    input_df.loc[0, 'LoanAmount'] = loan_amount
    input_df.loc[0, 'Loan_Amount_Term'] = 360  # default 360 months
    input_df.loc[0, 'Credit_History'] = 1 if credit_history=="Yes" else 0
    input_df.loc[0, 'TotalIncome'] = app_income + 0
    
    # Fill categorical using one-hot encoding columns
    input_df.loc[0, 'Self_Employed_Yes'] = 1 if employment_status=="Self_Employed" else 0
    input_df.loc[0, 'Property_Area_Semiurban'] = 1 if property_area=="Semiurban" else 0
    input_df.loc[0, 'Property_Area_Urban'] = 1 if property_area=="Urban" else 0
    
    # Fill other columns with 0
    for col in input_df.columns:
        if pd.isna(input_df.loc[0,col]):
            input_df.loc[0,col] = 0
    return input_df

input_data = prepare_input()

# ----------------- Prediction Button -----------------
if st.button("Check Loan Eligibility"):
    
    # Train SVM on full dataset
    X = df_encoded.drop(['Loan_ID','Loan_Status'],axis=1)
    y = df_encoded['Loan_Status'].map({'Y':1,'N':0})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Select kernel
    if kernel_choice=="Linear SVM":
        model = SVC(kernel='linear', probability=True)
    elif kernel_choice=="Polynomial SVM":
        model = SVC(kernel='poly', degree=3, probability=True)
    else:
        model = SVC(kernel='rbf', probability=True)
    
    # Fit model
    model.fit(X_scaled, y)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    conf_score = model.decision_function(input_scaled)[0]  # margin as confidence
    prob = model.predict_proba(input_scaled)[0][pred]  # probability for predicted class
    
    # ----------------- Display Results -----------------
    if pred==1:
        st.success(f"Loan Approved! (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f"Loan Rejected! (Confidence: {prob*100:.2f}%)")
    
    st.write(f"Kernel used: **{kernel_choice}**")
    
    # Business explanation
    if pred==1:
        st.info("Based on credit history and income pattern, the applicant is likely to repay the loan.")
    else:
        st.info("Based on credit history and income pattern, the applicant is unlikely to repay the loan.")
