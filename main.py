import zipfile
import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

extract_dir = './roof_data'

# Step 2: Create a data generator
class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=8, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_list = self._get_file_list()
        self.on_epoch_end()

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self._load_data(batch_files)
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)

    def _load_data(self, batch_files):
        images = []
        masks = []
        for json_path in batch_files:
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_path = os.path.join(os.path.dirname(json_path), data['image']['file_name'])
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image / 255.0)  # Normalize the image
                        # Create an empty mask
                        mask = np.zeros((data['image']['height'], data['image']['width']), dtype=np.uint8)
                        for annotation in data['annotations']:
                            rle = annotation['segmentation']
                            binary_mask = maskUtils.decode(rle)
                            mask = np.maximum(mask, binary_mask)
                        masks.append(mask / 255.0)  # Normalize the mask
                    else:
                        print(f"Failed to read image: {image_path}")
                else:
                    print(f"File does not exist: {image_path}")
        return np.array(images), np.array(masks)

# Step 3: Initialize data generators
train_data_dir = os.path.join(extract_dir, 'train')
val_data_dir = os.path.join(extract_dir, 'valid')

train_generator = DataGenerator(train_data_dir, batch_size=4, shuffle=True)
val_generator = DataGenerator(val_data_dir, batch_size=4, shuffle=False)

# Step 4: Define the model (U-Net example)
def unet_model(input_size=(1024, 1024, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet_model()

# Step 5: Train the model using the data generators
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # Adjust the number of epochs as needed
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Step 6: Validate the model
# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Save the retrained model
model.save('path/to/save/retrained_model.h5')