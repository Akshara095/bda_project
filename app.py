from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained logistic regression model
with open('logistic_regression_model_params.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Retrieve input from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Prepare data for prediction
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        
        # Predict and interpret result
        prediction = model.predict(input_data)
        result = 'Heart disease detected' if prediction == 1 else 'No heart disease detected'
    except Exception as e:
        result = f"An error occurred: {e}"
    
    return render_template('prediction.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
