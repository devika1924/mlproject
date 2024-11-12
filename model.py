import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('diabetes.csv') #diabetes dataset

data = data.dropna() #removes the rows that contains null values

# Features and Target
X = data[['Glucose', 'BloodPressure', 'BMI']] 
y = data['Outcome']  


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'model.joblib')

# # Print the accuracy to check the model's performance
# accuracy = model.score(X_test, y_test)
# print(f'Accuracy: {accuracy}')
