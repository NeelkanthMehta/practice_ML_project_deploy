import pickle
import numpy as np
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # request_data = request.get_json(force=True)
    # age = request_data['age']
    # salary = request_data['salary']
    input_value = [int(x) for x in request.form.values()]
    input_data = [np.array(input_value)]
    y_pred = model.predict(input_data)[0]
    y_pred_prob = model.predict_proba(input_data)[0][1]
    # return f"the prediction is  {y_pred:d}: {y_pred_prob:.2f}"
    return render_template('index.html', prediction=f'prediction: {y_pred}; probability: {100 * y_pred_prob:.2f}%')


if __name__ == '__main__':
    app.run(debug=True)
