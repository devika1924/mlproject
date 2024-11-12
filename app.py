from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the request(FORM)
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])

    # Create a DataFrame with the correct feature names
    features = pd.DataFrame([[feature1, feature2, feature3]], columns=['Glucose', 'BloodPressure', 'BMI'])

    # Make predictions using the loaded model
    prediction = model.predict(features)

    return render_template('predict.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
