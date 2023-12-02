from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory

import numpy as np
import os


# the images are in this format < attractiveness_level >_<acquisition_id>.jpg take att level as label
class DataPreprocess:
    """
    Data Preprocess class
    """

    def __init__(self, data_dir, img_height, img_width, batch_size):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def get_train_data(self):
        """
        :return:
        """
        train_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            # validation_split=0.2
        )
        data_dir = self.data_dir + '/training'
        train_generator = train_datagen.flow_from_directory(data_dir,
                                                            target_size=(self.img_height, self.img_width),
                                                            batch_size=self.batch_size, subset='training')
        return train_generator

    def get_validation_data(self):
        """
        :return:
        """
        validation_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1. / 255,
            validation_split=0.2
        )
        data_dir = self.data_dir + '/validation'

        validation_generator = validation_datagen.flow_from_directory(data_dir,
                                                                      target_size=(self.img_height, self.img_width),
                                                                      batch_size=self.batch_size, subset='validation')
        return validation_generator

    def get_test_data(self):
        """
        :return:
        """
        test_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1. / 255
        )
        data_dir = self.data_dir + '/test'

        test_generator = test_datagen.flow_from_directory(data_dir, target_size=(self.img_height, self.img_width),
                                                          batch_size=self.batch_size, shuffle=False)
        return test_generator

    def get_test_data_from_dir(self, test_dir):
        """
        :param test_dir:
        :return:
        """
        test_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1. / 255
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            shuffle=False
        )
        return test_generator
