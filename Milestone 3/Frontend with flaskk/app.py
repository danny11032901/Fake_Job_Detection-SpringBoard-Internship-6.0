from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("saved_model/fake_job_model.pkl")
vectorizer = joblib.load("saved_model/tfidf_vectorizer.pkl")

LOG_FILE = "predictions_log.csv"

# Create CSV file if not exists
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["job_description", "prediction", "confidence", "timestamp"])
    df.to_csv(LOG_FILE, index=False)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    job_desc = request.form["job_description"]

    # Transform input
    features = vectorizer.transform([job_desc])

    # Prediction + probability
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred] * 100

    label = "Fake Job" if pred == 1 else "Real Job"

    # --- Log to CSV ---
    log_row = pd.DataFrame([{
        "job_description": job_desc,
        "prediction": label,
        "confidence": f"{prob:.2f}%",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    log_row.to_csv(LOG_FILE, mode='a', header=False, index=False)

    return render_template("result.html",
                           prediction=label,
                           confidence=f"{prob:.2f}%")


@app.route("/history")
def history():
    df = pd.read_csv(LOG_FILE)
    return render_template("history.html", tables=df.to_dict(orient="records"))
    

if __name__ == "__main__":
    app.run(debug=True)
