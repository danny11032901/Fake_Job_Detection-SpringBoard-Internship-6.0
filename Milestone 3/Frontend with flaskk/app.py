from flask import Flask, render_template, request
import joblib, sqlite3
from datetime import datetime

app = Flask(__name__)

# Load ML model + vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# -------------------------------------------------
# CREATE DATABASE AND TABLE IF NOT EXISTS
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect('job_predictions.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.close()

init_db()



# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')



# -------------------------------------------------
# PREDICTION ROUTE  (SAVES TO SQLITE3)
# -------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    job_desc = request.form['job_description'].strip()

    # Minimum check
    if not job_desc or len(job_desc.split()) < 5:
        return render_template('index.html', error="⚠ Please enter a detailed job description (min 5 words).")

    # Predict
    X = vectorizer.transform([job_desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # Label + confidence
    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    # -------------------------------------------------
    # SAVE TO SQLITE DATABASE
    # -------------------------------------------------
    conn = sqlite3.connect('job_predictions.db')
    conn.execute(
        'INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
        (job_desc, label, confidence)
    )
    conn.commit()
    conn.close()

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           description=job_desc)



# -------------------------------------------------
# HISTORY PAGE (FETCHES FROM SQLITE3)
# -------------------------------------------------
@app.route('/history')
def history():
    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.execute(
        'SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC'
    )
    records = cursor.fetchall()
    conn.close()

    return render_template('history.html', records=records)



# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)