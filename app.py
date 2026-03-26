from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mrp = float(request.form['Item_MRP'])
    scaled = scaler.transform([[mrp]])
    prediction = model.predict(scaled)

    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == "__main__":
    app.run()
