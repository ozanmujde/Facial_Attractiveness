# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import model
from data_preprocess import DataPreprocess
from custom_data_generator import CustomDataGenerator


def process_data():
    print("TensorFlow version:", tf.__version__)
    data_preprocess = DataPreprocess(
        data_dir="/Users/ozan/PycharmProjects/DeepLearningHW/SCUT",
        img_height=80,
        img_width=80,
        batch_size=32,
    )
    train_generator = data_preprocess.get_train_data()
    validation_generator = data_preprocess.get_validation_data()
    test_generator = data_preprocess.get_test_data()
    cnn = model.CNN(
        input_shape=(80, 80, 3),
        num_classes=8,
        train_dataset=train_generator,
        validation_dataset=validation_generator,
        test_dataset=test_generator,
    )
    print("test_generator", len(test_generator.classes))
    cnnModel = cnn.build_model()
    cnn.train_and_evaluate(cnnModel, epochs=50, batch_size=32)
    # cnn.tune_model(train_generator, validation_generator, epochs=10, batch_size=32)
    # cnn.build_model(train_generator, validation_generator, epochs=10, batch_size=32)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    process_data()
