import pandas as pd
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import os

import matplotlib.pyplot as plt
import argparse

from data_generator_obsprog import ObservationProgramGenerator
import sys
sys.path.append("..")
from observation_program import ObservationProgram

class LSTMTrainer:
    def __init__(self, datagen, val_datagen, input_dim, save_location, target_columns=["ra", "decl"]):
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        self.train_data = datagen
        self.val_data = val_datagen
        self.n_input_dimensions = input_dim

        self.target_columns = target_columns
        self.save_path = save_location if save_location[-1] != "/" else save_location[:-1]

        self.model = self.make_model()

    def make_model(self):
        input_layer = tf.keras.layers.Input((2, self.n_input_dimensions))
        x = tf.keras.layers.LSTM(64, return_sequences=True)(input_layer)

        output_layers = []
        for target in self.target_columns:
            output_layers.append(tf.keras.layers.Dense(1, name=f"{target}_output")(x))

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layers)
        model.compile(
            optimizer="sgd", loss={f"{target}_output": "mse" for target in self.target_columns}
        )

        print(model.summary())
        return model

    def train_model(self, epochs):
        history = self.model.fit(
            self.train_data, validation_data=self.val_data, epochs=epochs).history
        return pd.DataFrame(history)

    def eval_metrics_plots(self, history):
        epochs = range(len(history))
        print(history)
        for target in self.target_columns:
            loss_train = history[f"{target}_output_loss"]
            loss_val = history[f"val_{target}_output_loss"]

            plt.plot(epochs, loss_train, label=f"{target} Train")
            plt.plot(epochs, loss_val, label=f"{target} Validation")

        plt.legend()
        plt.ylabel("MSE")
        plt.title("Loss History")
        plt.xlabel("Epoch")

        plt.savefig(f"{self.save_path}/loss_plot.png")

    def __call__(self, epochs):
        history = self.train_model(epochs=epochs)
        self.eval_metrics_plots(history)
        history.to_csv(f"{self.save_path}/model_history.csv")
        self.model.to_h5(f"{self.save_path}/model_weights.h5")


if __name__ == "__main__":

    default_output = os.path.abspath("../../results/lstm__generator_test/")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--out_path", default=default_output)
    argparser.add_argument("-e", "--epochs", type=int, default=1)

    args = argparser.parse_args()

    allowed_variables = ["seeing","clouds","lst","az","alt","zd","ha","airmass","sun_ra","sun_decl","sun_az","sun_alt","sun_zd","sun_ha","moon_ra","moon_decl","moon_az","moon_alt","moon_zd","moon_ha","moon_airmass","moon_phase","moon_illu","moon_Vmag","moon_angle","sky_mag","fwhm", "teff","exposure_time", "slew"]
    default_obsprog = "../train_configs/default_obsprog.conf"
    obsprog = ObservationProgram(config_path=default_obsprog, duration=2)
    datagen = ObservationProgramGenerator(obsprog)

    train = ObservationProgramGenerator(obsprog, included_variables=allowed_variables)
    val = ObservationProgramGenerator(obsprog, included_variables=allowed_variables)
    LSTMTrainer(
        datagen=train,
        val_datagen=val,
        input_dim=len(allowed_variables),
        save_location=args.out_path
    )(args.epochs)
