# Customer-churn-prediction-mlops3 🧑🏻
This project aims to predict customer churn using machine learning techniques. Customer churn is the phenomenon where customers stop using a company's products or services. By predicting which customers are likely to churn, businesses can take proactive measures to retain them, thereby improving customer retention and profitability.

# Project Overview
The objective of this project is to build a machine learning pipeline using ZenML that predicts whether a customer will churn based on various features such as account length, international plan, voice mail plan, call details, and other customer-related metrics. The pipeline includes data ingestion, data cleaning, model training, model evaluation, and deployment. The deployed model can be used to make real-time predictions through a Streamlit web application.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers us to build and deploy machine learning pipelines in multiple ways like:

- By integrating with tools like [MLflow](https://mlflow.org/) for deployment, tracking and more
- By allowing you to build and deploy your machine learning pipelines easily

## :snake: Python Requirements

Let's jump into the Python packages needed. Within the Python environment of your choice, run:

```bash
pip install -r requirements.txt
```
Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows us to observe the stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), and you must install the optional dependencies for the ZenML server:

```bash
pip install "zenml["server"]"
zenml init
zenml up
```
Installing mlflow integrations using ZenML:

```bash
zenml integration install mlflow -y
```

## File Structure:
![Screenshot 2024-06-10 084541](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/7aa190d4-70fa-4f03-8be5-b28bf924d8bc)

## Pipeline Development Process 🚀:

### 1. EDA -
![output](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/60c87c98-b69a-4bd8-97b3-89366ef50ade)

![output1](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/ec1efeb1-cfff-44a5-8915-9c8f436dcab4)

![output2](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/fcac3768-73e6-47aa-837c-6ea5d3b3d252)

![output3](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/0be7e300-dc65-4751-8ca9-a04a31279118)

![output4](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/8e0ba75d-f457-47e1-a604-765a89afc9e7)

### 2. Creation of a Blueprint of the classes -
   
   Steps > ingest_data.py , clean_data.py , model_train.py , evaluation.py

### 3. Data Cleaning -
   
   data_cleaning.py > -DataPreprocess,  -DataDivision

### 4. Model Development -
   
   Building the model on Train & Test datasets.

### 5. Defining Evaluation metrics -
   
   src > evaluation.py - defining MSE , RMSE , R2 Score

### 6. Training pipeline -
   - `ingest_data`: This step will ingest the data and create a `DataFrame`.
   - `clean_data`: This step will clean the data and remove the unwanted columns.
   - `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
   - `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.
     ![Screenshot 2024-06-10 080217](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/8bdf833a-e3f4-4527-9100-5381a1cdbecb)


The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

### 7. Deployment Pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.
![Screenshot 2024-06-10 080257](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/39ceaca6-76fc-4991-93a2-a8454cba7f74)


### 8. Inference Pipeline

This inference pipeline allows you to use the deployed model to make predictions on new customer data in real-time. 

![Screenshot 2024-06-10 080237](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/b389a6b9-79d6-4504-bfe5-13c9985fa85d)

### 9. Streamlit Application 

Streamlit Application Setup:

- Create a Streamlit application to serve the model :
  We design the UI to allow users to input features such as account length, international plan, voice mail plan, call details, and other metrics.
  
- Loading the Deployed Model :
  Use ZenML's prediction service loader to load the deployed model.
  Integrate the model prediction functionality into the Streamlit app.

- Making Predictions:
  The user inputs customer data through the Streamlit UI.
  The app sends the data to the deployed model, which returns the churn prediction.
  The prediction is displayed to the user in the app.

![Screenshot 2024-06-10 080523](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/761a3332-6a4a-4a28-ac5e-a7a9e69775a2)

![Screenshot 2024-06-10 080014](https://github.com/Keerthanareddy95/Customer-churn-prediction-mlops3/assets/123613605/cc4af162-b255-4d88-940d-0a34d74978a4)





