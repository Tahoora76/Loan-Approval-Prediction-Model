# Loan-Approval-Prediction-Model
Overview
The Loan Approval Prediction System is a machine learning-based project that predicts whether a loan application will be approved based on various input features. This project covers all stages of the ML pipeline, including data preprocessing, feature engineering, model building, and evaluation.

Features
End-to-End Pipeline: Data preprocessing, feature selection, model training, and evaluation.
Machine Learning Model: Random Forest Classifier for robust and interpretable predictions.
Data Insights: Visualizations like heatmaps and distribution plots.
Scalable Code: Modular structure for seamless integration into applications.
Local Testing: Test the model on new datasets with ease.
Dataset
Input Features:
loan_id - Unique identifier for each loan.
no_of_dependents - Number of dependents of the applicant.
education - Education level of the applicant (Graduate/Not Graduate).
self_employed - Employment type (Yes/No).
income_annum - Applicant's annual income.
loan_amount - Amount of the requested loan.
loan_term - Loan repayment period in months.
cibil_score - Applicant's credit score.
residential_assets_value - Value of residential assets owned.
commercial_assets_value - Value of commercial assets owned.
luxury_assets_value - Value of luxury assets owned.
bank_asset_value - Total value of the applicant's bank savings.
Output:
loan_status - Approval status (Approved/Not Approved).
Project Workflow
Data Preprocessing:

Handle missing values.
Encode categorical features.
Normalize numerical features.
Model Training:

Split data into train and test sets.
Train a Random Forest Classifier.
Evaluate using accuracy, confusion matrix, and classification report.
Testing:

Load the trained model and test with new datasets.
