from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model("face_spoof_model.keras")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") 
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    print("Raw Prediction Score:",prediction[0][0])

    if prediction[0][0] >= 0.5:
        result = "SPOOF"
    else:
        result = "REAL"

    return f"<h1>Prediction Result: {result}</h1><a href='/'>Go Back</a>"

if __name__ == "__main__":
    app.run(debug=True)
    