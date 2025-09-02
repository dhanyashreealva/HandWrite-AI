from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Image uploaded
    img = Image.open(file).convert('L').resize((28, 28))  # grayscale + resize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = np.argmax(model.predict(img_array), axis=1)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
