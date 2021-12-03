import os
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


path = os.path.dirname(__file__)

TRAINING_DIR = path + "/Image"
training_datagen = ImageDataGenerator(
	rescale=1. / 255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

VALIDATION_DIR = path + "/validation"
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150, 150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150, 150),
	class_mode='categorical'
)
model = tf.keras.models.Sequential([
	# Note the input shape is the desired size of the image 150x150 with 3 bytes color
	# This is the first convolution
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
	tf.keras.layers.MaxPooling2D(2, 2),
	# The second convolution
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	# The third convolution
	tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	# The fourth convolution
	tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	# Flatten the results to feed into a DNN
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dropout(0.5),
	# 512 neuron hidden layer
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=1)


def predict(img_path):
	labels = ['glass', 'metal', 'organic', 'plastic']
	img = image.load_img(img_path, target_size=(150, 150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	images = np.vstack([x])
	classes = model.predict(images, batch_size=10)
	return labels[np.argmax(classes)]


if __name__ == '__main__':
	predict('C:/Users/LENOVO/PycharmProjects/hackathon/Image/Glass/glass7.jpg')
