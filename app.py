from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import os
import csv
from datetime import datetime
from werkzeug.utils import secure_filename

from face_preprocessing import preprocess_image

app = Flask(__name__)

# =====================================
# PATH SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "face_spoof_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CSV_FILE = os.path.join(BASE_DIR, "detection_history.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================

print("Loading Face Spoof Detection Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =====================================
# CREATE CSV FILE
# =====================================

def create_csv():

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Image",
                "Result",
                "Score"
            ])


create_csv()


# =====================================
# SAVE PREDICTION HISTORY
# =====================================

def save_history(filename, result, score):

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            result,
            f"{score:.4f}"
        ])


# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================
# PREDICTION
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if not file or not file.filename:

        return "No image file was uploaded", 400

    filename = secure_filename(file.filename)

    if not filename:

        return "Invalid image filename", 400

    # Avoid overwriting images with the same filename
    unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + filename

    image_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(image_path)

    img = cv2.imread(image_path)

    try:

        model_input = preprocess_image(img)

    except ValueError as exc:

        return str(exc), 400

    # Model prediction
    prediction = model.predict(model_input, verbose=0)

    score = float(prediction[0][0])

    print("Raw Prediction Score:", score)

    # Classification
    if score >= 0.5:

        result = "SPOOF"

    else:

        result = "REAL"

    # Save prediction in CSV
    save_history(unique_name, result, score)

    # Display prediction result
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Spoof Detection Result</title>
        <style>
            body {{
                font-family: Arial;
                text-align: center;
                padding: 40px;
                background: #eaf2ff;
            }}

            h1 {{
                color: {"#ef4444" if result == "SPOOF" else "#22c55e"};
            }}

            a {{
                display: inline-block;
                margin: 15px;
                padding: 12px 20px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                border-radius: 7px;
            }}
        </style>
    </head>

    <body>

        <h1>Prediction Result: {result}</h1>

        <h3>Spoof Score: {score:.4f}</h3>

        <br>

        <a href="/">GO BACK</a>

        <a href="/dashboard">OPEN DASHBOARD</a>

    </body>
    </html>
    """


# =====================================
# DASHBOARD
# =====================================

@app.route("/dashboard")
def dashboard():

    records = []

    real_count = 0
    spoof_count = 0

    # Read prediction history
    if os.path.exists(CSV_FILE):

        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                records.append(row)

                if row["Result"] == "REAL":

                    real_count += 1

                elif row["Result"] == "SPOOF":

                    spoof_count += 1

    # Calculate statistics
    total = real_count + spoof_count

    if total > 0:

        real_percent = (real_count / total) * 100

        spoof_percent = (spoof_count / total) * 100

    else:

        real_percent = 0

        spoof_percent = 0

    # Latest records first
    records.reverse()

    return render_template(
        "dashboard.html",
        total=total,
        real_count=real_count,
        spoof_count=spoof_count,
        real_percent=real_percent,
        spoof_percent=spoof_percent,
        records=records
    )


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":

    app.run(debug=True)