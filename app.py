import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

PATH = "./dataset"

data_dir_list = os.listdir(PATH)
img_rows, img_cols = 224, 224
num_channel = 3
num_epoch = 100
batch_size = 32

img_data_list = []
classes_names_list = []
target_column = []

for dataset in data_dir_list:
    classes_names_list.append(dataset)
    img_list = os.listdir(PATH + "/" + dataset)
    for img in img_list:
        input_img = cv2.imread(PATH + "/" + dataset + "/" + img)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        img_data_list.append(input_img_resize)
        target_column.append(dataset)

img_data = np.array(img_data_list)
target_column = np.array(target_column)

label_encoder = LabelEncoder()
target_column_encoded = label_encoder.fit_transform(target_column)
target_column_onehot = to_categorical(target_column_encoded)

img_data, target_column_onehot = shuffle(img_data, target_column_onehot, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(img_data, target_column_onehot, test_size=0.2, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, num_channel))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(len(classes_names_list), activation='softmax'))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=num_epoch,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

model.load_weights('best_model.keras')

accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy[1])