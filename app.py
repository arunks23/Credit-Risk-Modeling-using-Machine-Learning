
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
input_method = st.sidebar.radio("Choose the data input method:", ("Home Page" , "Upload CSV File"))

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
        st.write(" The dashboard displays the risk predictions, categorizing customers based on their likelihood of defaulting on credit card payments. The predictions are categorized as follows:")
        st.subheader("p1")
        st.write("Best candidate for credit card issuance, lowest risk.")
        st.subheader("p2")
        st.write("Second best candidate, low risk.")
        st.subheader("p3")
        st.write("Third best candidate, moderate risk")
        st.subheader("p4")
        st.write(" Least suitable candidate, highest risk.")
    with col2:
        st.header("Risk appetite")
        st.subheader("Low")
        st.write("Targets already achieve,p1")
        st.subheader("High")
        st.write("Target are far away , p1, p2, p3")
        st.subheader("Severly High")
        st.write("Target are very far away , p1, p2, p3, p4")


if input_method == "Upload CSV File":
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    # Handle file upload
    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        if st.button('Predict'):
            data.loc[data['EDUCATION'] == 'SSC', ['EDUCATION']] = 1
            data.loc[data['EDUCATION'] == '12TH', ['EDUCATION']] = 2
            data.loc[data['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 3
            data.loc[data['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 3
            data.loc[data['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 4
            data.loc[data['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1
            data.loc[data['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']] = 3
            data['EDUCATION'].value_counts()
            data['EDUCATION'] = data['EDUCATION'].astype(int)
            final_df = pd.get_dummies(data, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])
            answer = load_model.predict(final_df)
            data['predictions'] = answer
            for i in data['predictions']:
                if i == 0:
                    data['predictions'] = data['predictions'].replace(i,'p1')
                elif i == 2:
                    data['predictions'] = data['predictions'].replace(i,'p2')
                elif i == 3:
                    data['predictions'] = data['predictions'].replace(i,'p3')
                elif i == 4:
                    data['predictions'] = data['predictions'].replace(i,'p4')
            st.write(data)
