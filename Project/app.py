from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load the standard scaler and model from pickle files
scaler = pickle.load(open("/config/workspace/Project/Model/standard_scaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/Project/Model/linear_regression_model.pkl", "rb"))

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        num_bedrooms = int(request.form.get("num_bedrooms"))
        num_bathrooms = float(request.form.get('num_bathrooms'))
        square_footage = float(request.form.get('square_footage'))
        age_of_house = float(request.form.get('age_of_house'))
        proximity_to_city_center = float(request.form.get('proximity_to_city_center'))

        new_data = scaler.transform([[num_bedrooms, num_bathrooms, square_footage, age_of_house, proximity_to_city_center]])
        predict = model.predict(new_data)

        result = f'The predicted house price is ${predict[0]:,.2f}'

    return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
