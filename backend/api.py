import numpy as np
import os
import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib



class ConvolutionalNeuronalNetwork:

    def __init__(self):
        self.batch_size = 32
        self.img_width = 180
        self.img_height = 180
        self.num_classes = 2
        self.epochs = 100
        
        self.readDirectory()
        self.reserveDataForTraining()
        self.reserveDataForValidation()
        self.storageCacheImages()
        self.standardization()
        self.createModel()
        self.training()



    def readDirectory(self):
        self.data_dir = tf.keras.utils.get_file(  os.path.join( os.getcwd(),"dataset"),  origin=os.path.join( os.getcwd()), untar=False)
        self.data_dir = pathlib.Path(self.data_dir)

    def reserveDataForTraining(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names


    def reserveDataForValidation(self):
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)
            
    def storageCacheImages(self):
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def standardization(self):
        self.normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
        self.normalized_ds = self.train_ds.map(lambda x, y: (self.normalization_layer(x), y))

    def createModel(self):

      self.model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
      ])
      self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
      self.model.summary()

      
    def training(self):
      epochs=100
      self.history = self.model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs
    )

    def prediction(self):
        
        img = keras.preprocessing.image.load_img(
            os.path.join(os.getcwd(), "test1.jpg"), target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return {"label": format(self.class_names[np.argmax(score)]), "percentage": 100 * np.max(score)}




convolutionNetwork = ConvolutionalNeuronalNetwork()



app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
             return jsonify({"error": "no hay"})
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "no hay"})

        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return jsonify(convolutionNetwork.prediction())

app.run()