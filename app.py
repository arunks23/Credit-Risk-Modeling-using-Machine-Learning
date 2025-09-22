import streamlit as st
import pandas as pd
import pickle
import xgboost

# Load the trained model
try:
    with open("best_model.pkl", 'rb') as file:
        load_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Set the title of the page
st.title("Welcome to the Credit Card Risk Prediction Dashboard")

# Sidebar for data input method selection
input_method = st.sidebar.radio("Choose the data input method:", ("Home Page", "Upload CSV File"))

if input_method == "Home Page":
    st.write("Thank you for visiting our Credit Card Risk Prediction Dashboard. This tool is designed to help financial institutions and credit analysts assess the risk associated with issuing credit cards to potential customers.")
    st.header("How It Works")
    st.subheader("1 Data Input:")
    st.write("You can upload customer data in CSV format to the dashboard. This data should include various features related to the customer's financial history and personal details.")
    st.subheader("2 Prediction:")
    st.write("Once the CSV file is uploaded, our trained machine learning model processes the data and provides a risk prediction for each customer.")
    st.subheader("3 Results")
    col1, col2 = st.columns(2)
    with col1:
        st.write("The dashboard displays the risk predictions, categorizing customers based on their likelihood of defaulting on credit card payments. The predictions are categorized as follows:")
        st.subheader("p1")
        st.write("Best candidate for credit card issuance, lowest risk.")
        st.subheader("p2")
        st.write("Second best candidate, low risk.")
        st.subheader("p3")
        st.write("Third best candidate, moderate risk.")
        st.subheader("p4")
        st.write("Least suitable candidate, highest risk.")
    with col2:
        st.header("Risk appetite")
        st.subheader("Low")
        st.write("Targets already achieved, p1")
        st.subheader("High")
        st.write("Targets are far away, p1, p2, p3")
        st.subheader("Severely High")
        st.write("Targets are very far away, p1, p2, p3, p4")

st.header("Make sure to use a dataset having columns of the following type:")
df = pd.read_csv("final_df.csv")
st.write(df.head())

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if st.button('Predict'):
            # Encode EDUCATION
            data.loc[data['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
            data.loc[data['EDUCATION'] == '12TH', 'EDUCATION'] = 2
            data.loc[data['EDUCATION'].isin(['GRADUATE', 'UNDER GRADUATE', 'PROFESSIONAL']), 'EDUCATION'] = 3
            data.loc[data['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
            data.loc[data['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
            data['EDUCATION'] = data['EDUCATION'].astype(int)

            # One-hot encode categorical features
            final_df = pd.get_dummies(data, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

            # Align final_df with model features
            train_features = load_model.get_booster().feature_names
            for col in train_features:
                if col not in final_df.columns:
                    final_df[col] = 0  # add missing columns
            final_df = final_df[train_features]  # reorder columns

            # Make predictions
            answer = load_model.predict(final_df)

            # Map predictions to labels
            mapping = {0: 'p1', 1: 'p2', 2: 'p3', 3: 'p4'}
            data['predictions'] = [mapping[i] for i in answer]

            st.write(data)
