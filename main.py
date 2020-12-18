import numpy as np
import os
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

DATADIR = "spiral"

CATEGORIES = ["Healthy", "Effected"]

IMG_SIZE = 50

training_data = []

for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)


def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass


create_training_data()

random.shuffle(training_data)

print(training_data)

X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


X = X/255.0


model = Sequential()


model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))


model.add(Dense(128))
model.add(Activation("relu"))


model.add(Dense(2))
model.add(Activation("softmax"))


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


y = np.array(y)
history = model.fit(X, y, batch_size=32, epochs=40, validation_split=0.1)
import joblib
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
joblib.dump(scaler,'scaler.pkl')

model.save("CNN.h5")

