import random
from keras import layers, models, optimizers, initializers, regularizers
from keras import losses
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.initializers.initializers_v2 import Initializer


class CNN(kt.HyperModel):
    """
    Convolutional Neural Network
    """

    def __init__(
        self, input_shape, num_classes, train_dataset, validation_dataset, test_dataset
    ):
        super().__init__()
        seed = 42
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        # self.create_model(input_shape)

    def __call__(self, shape, dtype=None, **kwargs):
        # returns a tensor of shape `shape` and dtype `dtype`
        # containing values drawn from a distribution of your choice.
        return tf.random.uniform(shape=shape, dtype=dtype)

    def create_model(self, hp):
        """

        :param input_shape: tuple
        :param hp: HyperParameters
        """
        model = models.Sequential()
        kernel_size = hp.Choice("kernel_size", values=[1, 3, 5])
        first_filter_size = hp.Choice("first_filter_size", values=[32, 64, 128])
        filter_size = hp.Choice("filter_size", values=[64, 128, 256])
        l2_learning_rate = hp.Float(
            "l2_learning_rate",
            min_value=1e-4,
            max_value=1e-2,
            sampling="LOG",
            default=1e-3,
        )
        model.add(
            layers.Conv2D(
                first_filter_size, kernel_size, activation="relu", input_shape=self.input_shape,
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(filter_size, kernel_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(filter_size, kernel_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                hp.Int("units", min_value=32, max_value=256, step=32), activation="relu"
            )
        )
        model.add(layers.BatchNormalization())
        # add kernel_regularizer
        model.add(layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1)))
        model.add(layers.Dense(1))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-3,
            max_value=1e-1,
            sampling="LOG",
            default=1e-2,
        )
        model.compile(
            optimizer=optimizers.legacy.Adam(learning_rate=hp_learning_rate),
            loss=hp.Choice('loss_function',["mse", "mae"]),
            metrics=["MAE"],
        )
        model.summary()
        return model

    def get_best_epoch(self, hp):
        model = self.create_model(hp)
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=5)]
        history = model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=50,
            batch_size=128,
            callbacks=callbacks,
        )
        val_loss_per_epoch = history.history["val_loss"]
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print(f"Best epoch: {best_epoch}")
        return best_epoch

    def get_best_trained_model(self, hp):
        best_epoch = self.get_best_epoch(hp)
        model = self.create_model(hp)
        model.fit(self.train_dataset, batch_size=32, epochs=int(best_epoch * 1.2))
        return model

    def build_model(self):
        """
        :return:
        """
        initializer = initializers.RandomNormal()
        model = models.Sequential()
        model.add(
            layers.Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=self.input_shape,
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(0.001),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(
                64,
                (5, 5),
                activation="relu",
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(0.001),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(
                64,
                (5, 5),
                activation="relu",
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(0.001),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                32,
                activation="relu",
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(0.001),
            )
        )
        model.add(layers.BatchNormalization())
        # add kernel_regularizer
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        model.compile(
            optimizer=optimizers.legacy.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=["mae"],
        )
        model.summary()
        return model

    def tune_model(self, train_dataset, validation_data, epochs, batch_size):
        tuner = kt.BayesianOptimization(
            self.create_model,
            objective="val_loss",
            max_trials=epochs,
            directory="tuner_logs",
            project_name="DL_HW_1",
        )
        # stop_early = EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(train_dataset, epochs=epochs, validation_data=validation_data)
        tuner.results_summary()
        # Get the optimal hyperparameters

        top_n = 4
        best_hps = tuner.get_best_hyperparameters(top_n)[0]
        print(
            f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')} and the best kernel size is {best_hps.get('kernel_size')}.
        """
        )
        # best_models = []
        # for hp in best_hps:
        #     model = self.get_best_trained_model(hp)
        #     model.evaluate(validation_data)
        #     best_models.append(model)
        # best_models = tuner.get_best_models(top_n)

        return best_hps

    def train_and_evaluate(self, model, epochs, batch_size, model_name="model.h5"):
        """
        :param model:
        :param epochs:
        :param batch_size:
        :param model_name:
        """
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=5)]
        history = model.fit(
            self.train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
        )
        print(history.history.keys())
        mae_history = history.history["val_loss"]
        train_maes = history.history["loss"]
        plt.plot(mae_history)
        plt.plot(train_maes)
        plt.title("MAE Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend(["Validation MAE", "Train MAE"])
        plt.show()
        model.evaluate(self.test_dataset)
        predictions = model.predict(self.test_dataset)
        rounded_predictions = np.round(predictions)
        # accuracy = (rounded_predictions == self.test_dataset.labels).count() / len(
        #     self.test_dataset.labels
        # )
        count = 0
        for i in range(len(rounded_predictions)):
            if rounded_predictions[i] == self.test_dataset.labels[i]:
                count += 1
        accuracy = count / len(rounded_predictions)
        print("Accuracy: ", accuracy)
        model.save(model_name)
