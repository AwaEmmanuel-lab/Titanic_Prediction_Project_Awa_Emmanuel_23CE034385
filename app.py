from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load saved model
with open("./model/titanic_survival_model.pkl", "rb") as file:
    model, scaler, encoder = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form["pclass"])
    sex = encoder.transform([request.form["sex"]])[0]
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    fare = float(request.form["fare"])

    data = np.array([[pclass, sex, age, sibsp, fare]])
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]
    result = "Survived" if pred == 1 else "Did Not Survive"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
