import os 
print("CURRENT DIRECTORY:",os.getcwd())
print("FILES AVAILABLE:",os.listdir())
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "Sales Prediction API Running"

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        mrp = data['Item_MRP']
    else:
        mrp = request.args.get('Item_MRP')

    scaled = scaler.transform([[float(mrp)]])
    prediction = model.predict(scaled)

    return {'prediction': float(prediction[0])}

if __name__ == '__main__':
    app.run()
