import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("End to End Customer Churn Prediction Pipeline with ZenML")

    st.markdown(
        """ 
    #### Problem Statement 
    The objective here is to predict customer churn based on various features like account length, international plan, voice mail plan, call activities, etc. We will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict whether a customer will churn.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict whether a customer will churn based on the features listed below. You can input the values for these features and get a churn prediction. 
    | Feature                             | Description                                           | 
    | ----------------------------------- | ----------------------------------------------------- | 
    | account_length                      | The duration of the customer account in days          | 
    | international_plan                  | Whether the customer has an international plan (0/1)  | 
    | voice_mail_plan                     | Whether the customer has a voice mail plan (0/1)      | 
    | number_vmail_messages               | Number of voice mail messages                         | 
    | total_day_calls                     | Total number of day calls                             | 
    | total_eve_calls                     | Total number of evening calls                         | 
    | total_night_calls                   | Total number of night calls                           | 
    | total_intl_calls                    | Total number of international calls                   | 
    | number_customer_service_calls       | Number of calls to customer service                   | 
    | area_code_encoded                   | Encoded area code                                     | 
    | total_day_minutes                   | Total minutes of day calls                            | 
    | total_eve_minutes                   | Total minutes of evening calls                        | 
    | total_night_minutes                 | Total minutes of night calls                          | 
    | total_intl_minutes                  | Total minutes of international calls                  | 
    """
    )

    account_length = st.sidebar.slider("Account Length", min_value=0, max_value=500, value=100)
    international_plan = st.sidebar.selectbox("International Plan", [0, 1])
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", [0, 1])
    number_vmail_messages = st.sidebar.slider("Number of Voice Mail Messages", min_value=0, max_value=50, value=0)
    total_day_calls = st.sidebar.slider("Total Day Calls", min_value=0, max_value=200, value=100)
    total_eve_calls = st.sidebar.slider("Total Evening Calls", min_value=0, max_value=200, value=100)
    total_night_calls = st.sidebar.slider("Total Night Calls", min_value=0, max_value=200, value=100)
    total_intl_calls = st.sidebar.slider("Total International Calls", min_value=0, max_value=20, value=10)
    number_customer_service_calls = st.sidebar.slider("Number of Customer Service Calls", min_value=0, max_value=10, value=1)
    area_code_encoded = st.sidebar.slider("Area Code (Encoded)", min_value=0, max_value=1000, value=415)
    total_day_minutes = st.sidebar.slider("Total Day Minutes", min_value=0, max_value=400, value=200)
    total_eve_minutes = st.sidebar.slider("Total Evening Minutes", min_value=0, max_value=400, value=200)
    total_night_minutes = st.sidebar.slider("Total Night Minutes", min_value=0, max_value=400, value=200)
    total_intl_minutes = st.sidebar.slider("Total International Minutes", min_value=0, max_value=60, value=30)

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "account_length": [account_length],
                "international_plan": [international_plan],
                "voice_mail_plan": [voice_mail_plan],
                "number_vmail_messages": [number_vmail_messages],
                "total_day_calls": [total_day_calls],
                "total_eve_calls": [total_eve_calls],
                "total_night_calls": [total_night_calls],
                "total_intl_calls": [total_intl_calls],
                "number_customer_service_calls": [number_customer_service_calls],
                "area_code_encoded": [area_code_encoded],
                "total_day_minutes": [total_day_minutes],
                "total_eve_minutes": [total_eve_minutes],
                "total_night_minutes": [total_night_minutes],
                "total_intl_minutes": [total_intl_minutes],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "The churn prediction for the given customer details is: {}".format(pred)
        )

if __name__ == "__main__":
    main()
