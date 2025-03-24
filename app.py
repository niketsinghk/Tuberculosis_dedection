# app.py

from flask import Flask, request, render_template # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore

from PIL import Image # type: ignore


app = Flask(__name__)

# Load your trained TensorFlow model
model = keras.models.load_model(r"C:\Users\niket\Desktop\TuberculosisAI\TuberculosisAI\models\tb_detection_model.h5")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image = request.files['image']
        img = Image.open(image)

        # Preprocess the image (resize, normalize, etc.)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # Normalize pixel values
        img = img.reshape((1, 224, 224, 3))  # Reshape for model input

        # Make a prediction using the loaded model
        prediction = model.predict(img)
        class_label = "TB Positive" if prediction[0][0] > 0.5 else "TB Negative"

        return render_template('result.html', prediction=class_label)
    except Exception as e:

        # Center the exception message in the middle of the webpage
        error_message = str(e)
        return render_template('result.html', error_message=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

