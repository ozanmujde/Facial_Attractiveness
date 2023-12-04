from keras import layers
from keras import models
from keras import optimizers
from keras import losses
import keras_tuner as kt
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


class CNN(kt.HyperModel):
    """
    Convolutional Neural Network
    """

    def __init__(
        self, input_shape, num_classes, train_dataset, validation_dataset, test_dataset
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        # self.create_model(input_shape)

    def create_model(self, hp):
        """

        :param input_shape: tuple
        :param hp: HyperParameters
        """
        # TODO play with it
        model = models.Sequential()
        kernel_size = hp.Choice("kernel_size", values=[1, 3, 5])
        model.add(
            layers.Conv2D(
                32, kernel_size, activation="relu", input_shape=self.input_shape
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, kernel_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, kernel_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                hp.Int("units", min_value=32, max_value=256, step=32), activation="relu"
            )
        )
        model.add()
        # add kernel_regularizer
        model.add(layers.Dropout(0.25))
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
            loss="mse",
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
        model = models.Sequential()
        model.add(
            layers.Conv2D(32, (5, 5), activation="relu", input_shape=self.input_shape)
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation="leaky_relu"))

        # add kernel_regularizer
        # self.model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        model.compile(
            optimizer=optimizers.legacy.Adam(learning_rate=1e-3),
            loss="mae",
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
        # print(history.history.keys())
        # mae_history = history.history["val_loss"]
        # train_maes = history.history["loss"]
        # accuracy_history = history.history["val_accuracy"]
        # train_accuracy = history.history["accuracy"]
        # plt.plot(range(1, len(mae_history) + 1), mae_history)
        # plt.plot(range(1, len(train_maes) + 1), train_maes)
        # plt.xlabel("Epochs")
        # plt.ylabel("MAE")
        # # plt.show()
        # plt.plot(range(1, len(accuracy_history) + 1), accuracy_history)
        # plt.plot(range(1, len(train_accuracy) + 1), train_accuracy)
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.show()
        predictions = model.predict(self.test_dataset)
        rounded_predictions = np.round(predictions)
        mae = np.mean(abs(rounded_predictions - self.test_dataset.labels))
        print("MAE: ", mae)
        # accuracy = (rounded_predictions == self.test_dataset.labels).count() / len(
        #     self.test_dataset.labels
        # )
        count = 0
        for i in range(len(rounded_predictions)):
            if rounded_predictions[i] == self.test_dataset.labels[i]:
                count += 1
        accuracy = count / len(rounded_predictions)
        print(count)
        print("Accuracy: ", accuracy)
        model.save(model_name)
