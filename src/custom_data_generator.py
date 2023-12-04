import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def get_label(filename):
    """
    Convert the filename into a label.
    """
    score = filename.split("_")[0]  # assuming filename is SCORE_index.jpg
    return score


class CustomDataGenerator:
    def __init__(self, directory, batch_size, target_size):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.filenames = os.listdir(directory)
        self.num_files = len(self.filenames)

    def generate(self):
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=self.filenames, size=self.batch_size)
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = load_img(
                    self.directory + "/" + input_path, target_size=self.target_size
                )
                output = get_label(input_path)

                input = img_to_array(input)
                batch_input += [input]
                batch_output += [output]
            # Return a tuple of (input, output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
