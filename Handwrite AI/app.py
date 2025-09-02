from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/')
def home():
    return render_template("index.html")  # your frontend page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read and preprocess image
    img = Image.open(io.BytesIO(file.read())).convert('L')  # grayscale
    img = img.resize((28, 28))  # MNIST is 28x28
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # batch, height, width, channels

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
