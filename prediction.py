import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, request

CATEGORIES = ["Healthy", "Effected"]


def prepare(file):
    IMG_SIZE = 50
    new_array = cv2.resize(file, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


app = Flask(__name__)


def load_model():
    global model

    model = tf.keras.models.load_model("CNN.h5")


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        print("1")
        # user_image = request.files["file"]
        # image = cv2.imdecode(np.frombuffer(user_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread('image.png', 0)
        preparedImage = prepare(image)

        prediction = model.predict([preparedImage])
        print("6")
        prediction = list(prediction[0])

        return CATEGORIES[prediction.index(max(prediction))]


if __name__ == '__main__':
    load_model()  
    app.run(debug=True)
