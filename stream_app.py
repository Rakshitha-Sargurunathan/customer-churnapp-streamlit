import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and DictVectorizer
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Function to predict churn
def predict_churn(input_dict):
    X = dv.transform([input_dict])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    return bool(churn), float(y_pred)

# Function to visualize charts
def visualize_charts(df):
    st.subheader("Visualizing Charts")
    
    # Set style
    sns.set_style("whitegrid")

    # Bar chart for churn distribution based on contract type
    st.subheader("Churn Distribution by Contract Type")
    contract_churn_counts = df.groupby('contract')['churn'].value_counts().unstack()
    fig, ax = plt.subplots()
    contract_churn_counts.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=ax)
    ax.set_xlabel('Contract Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Bar chart for churn distribution based on gender
    st.subheader("Churn Distribution by Gender")
    gender_churn_counts = df.groupby('gender')['churn'].value_counts().unstack()
    fig, ax = plt.subplots()
    gender_churn_counts.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # Bar chart for churn distribution based on internet service type
    st.subheader("Churn Distribution by Internet Service Type")
    internet_churn_counts = df.groupby('internetservice')['churn'].value_counts().unstack()
    fig, ax = plt.subplots()
    internet_churn_counts.plot(kind='bar', stacked=True, color=['#9467bd', '#8c564b'], ax=ax)
    ax.set_xlabel('Internet Service Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # Bar chart for churn distribution based on payment method
    st.subheader("Churn Distribution by Payment Method")
    payment_churn_counts = df.groupby('paymentmethod')['churn'].value_counts().unstack()
    fig, ax = plt.subplots()
    payment_churn_counts.plot(kind='bar', stacked=True, color=['#e377c2', '#7f7f7f'], ax=ax)
    ax.set_xlabel('Payment Method', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    # Load data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    df.churn = (df.churn == 'yes').astype(int)

    # Sidebar options
    st.title("PREDICTING CUSTOMER CHURN")
    st.info("This web app is created to predict Customer Churn")
    
    
    #option = st.sidebar.selectbox("Select option", [" ","Predict Churn", "Visualize Charts", "Display Dataset"])
    option = st.sidebar.radio("CONTENTS", ["Home","Predict Churn", "Visualize Charts", "Display Dataset"])
    if option == "Home":
        st.write("""
    ## Welcome to our Customer Churn Prediction Web Application!

    **What does this app do?**

    This web app is designed to predict customer churn for telecom service providers. Customer churn, also known as customer attrition, refers to the phenomenon of customers ceasing their relationship with a company. By analyzing various customer attributes and behaviors, our app predicts the likelihood of a customer churning from the telecom service.

    **How does it work?**

    Using machine learning algorithms trained on historical customer data, our app analyzes factors such as contract type, monthly charges, internet service, and more to predict whether a customer is likely to churn or not. You can input specific details about a customer, and our app will provide you with a churn prediction along with a risk score.

    **Why is it important?**

    Predicting customer churn is crucial for telecom companies to retain their customers and maintain profitability. By identifying customers at risk of churning, companies can take proactive measures such as offering incentives or personalized services to prevent them from leaving.

    **How to use?**

    Simply select the "Predict Churn" option from the dropdown menu, input the relevant customer details, and click on the "Predict" button. Our app will then provide you with the churn prediction and risk score for the customer.

    Feel free to explore the other options in the dropdown menu to visualize churn distribution and display the dataset used for prediction.

    Start predicting customer churn now and optimize your business strategies!
    """)

    if option == "Predict Churn":
        st.empty()
        st.subheader("Predict Churn")
		
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
        partner= st.selectbox(' Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
        phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
        multiplelines = st.selectbox(' Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
        internetservice= st.selectbox(' Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox(' Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox(' Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
        contract= st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
        paymentmethod= st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
        monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
        totalcharges = tenure*monthlycharges
        output= ""
        output_prob = ""
        input_dict={
                    "gender":gender ,
                    "seniorcitizen": seniorcitizen,
                    "partner": partner,
                    "dependents": dependents,
                    "phoneservice": phoneservice,
                    "multiplelines": multiplelines,
                    "internetservice": internetservice,
                    "onlinesecurity": onlinesecurity,
                    "onlinebackup": onlinebackup,
                    "deviceprotection": deviceprotection,
                    "techsupport": techsupport,
                    "streamingtv": streamingtv,
                    "streamingmovies": streamingmovies,
                    "contract": contract,
                    "paperlessbilling": paperlessbilling,
                    "paymentmethod": paymentmethod,
                    "tenure": tenure,
                    "monthlycharges": monthlycharges,
                    "totalcharges": totalcharges
                }

        if st.button("Predict"):
                X = dv.transform([input_dict])
                y_pred = model.predict_proba(X)[0, 1]
                churn = y_pred >= 0.5
                output_prob = float(y_pred)
                output = bool(churn)
        st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

    elif option == "Visualize Charts":
        visualize_charts(df)

    elif option == "Display Dataset":
        st.subheader("Telco Customer Churn Dataset")
        st.dataframe(df)

if __name__ == '__main__':
    main()
