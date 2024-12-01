# Loan Approval Prediction System

This project is a machine learning-based system that predicts loan approvals using **Random Forest Classifier**. It analyzes applicant details, processes the data, and provides an accurate prediction on loan eligibility.

## Features:
- Predicts loan approval based on multiple applicant features.
- Real-time input support for testing new cases.
- Handles missing data with preprocessing techniques.
- Visualizations for data insights and feature correlations.

## Requirements

To set up the project and install the necessary dependencies, follow the steps below:

### Clone the repository:
```bash
git clone https://github.com/Tahoora76/Loan-Approval-Prediction-Model.git
cd Loan-Approval-Prediction-Model
```
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## How It Works
- Data Preprocessing
- Handles missing values.
- Encodes categorical variables.
- Scales numerical features.
## Model Training
- Uses a Random Forest Classifier.
- Trained on features like income, loan amount, CIBIL score, etc.
## Prediction
- Predicts loan approval (Approved or Rejected) based on applicant features.
## Evaluation
- Evaluates model performance using accuracy and classification metrics.
