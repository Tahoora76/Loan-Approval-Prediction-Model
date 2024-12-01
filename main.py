import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'loan_approval_dataset.csv'
data = pd.read_csv(file_path)

# Step 1a: Clean column names by stripping leading/trailing spaces
data.columns = data.columns.str.strip()

# Step 2: Initial exploration
print("Dataset Shape:", data.shape)
print("Dataset Info:")
print(data.info())
print("Missing Values:\n", data.isnull().sum())

# Step 3: Handle Missing Values
# Fill numerical columns with median and categorical with mode
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)  # Fill categorical missing with mode
    else:
        data[column].fillna(data[column].median(), inplace=True)  # Fill numerical missing with median

# Verify no missing values remain
print("\nMissing Values After Imputation:\n", data.isnull().sum())

# Step 4: Encode categorical features
encoder = LabelEncoder()
categorical_columns = ['education', 'self_employed']  # Specify categorical columns
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Step 5: Define Features and Target Variable
X = data.drop(['loan_id', 'loan_status'], axis=1)  # Drop irrelevant and target columns
y = data['loan_status'].apply(lambda x: 1 if x == 'Approved' else 0)  # Encode target variable

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Feature Importance (Optional - Visualize Feature Importance)
feature_importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Step 11: Save the model (Optional)
import joblib
joblib.dump(model, 'loan_approval_model.pkl')

# Step 12: Predict new data (Optional - Example)
# new_data = pd.DataFrame({...})  # Define new data here
# new_prediction = model.predict(new_data)
# print("New Data Prediction:", new_prediction)
