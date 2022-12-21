"""
A very basic lstm code to predict an action between two steps

"""
import pandas as pd
import numpy as np
import tqdm
import torch.nn.functional as F
import torch

from observation_generator import ObservationGenerator


class OfflineLstm(torch.nn.Module):
    def __init__(self, input_dim):
        super(OfflineLstm, self).__init__()
        self.lstm_layer = torch.nn.LSTM(input_dim, 64, dropout=0.2)
        self.out_layer = torch.nn.Linear(64, 2)

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(x)
        x = self.out_layer(final_hidden_state[-1])
        return x


class TrainOfflineLstm:
    def __init__(self, input_dim, optimizer, criterion, n_epochs):
        self.model = OfflineLstm(input_dim=input_dim)
        self.optimizer = optimizer(self.model.parameters(), lr=0.01)

        self.loss_criterion = criterion
        self.n_epochs = n_epochs

        self.loss_history = {"train_loss": {}, "val_loss": {}}

    def train(self, epoch, train_data, val_data):
        self.model.train()  # Put it in train mode
        n_batches = len(train_data)
        self.loss_history["train_loss"][epoch] = []
        self.loss_history["val_loss"][epoch] = []

        for batch_id, data in enumerate(train_data):
            self.optimizer.zero_grad()
            observation = data["observation"]
            if observation.size()[-1] != 0:
                output = self.model(observation)

                loss = self.loss_criterion(output, data["actions"])
                loss.backward()
                self.optimizer.step()

                if np.ceil(epoch / 4) % n_batches == 0:
                    self.loss_history["train_loss"][epoch].append(loss.item())
                    self.loss_history["val_loss"][epoch].append(self.test(val_data))
            else:
                break

    def test(self, val_data):
        self.model.eval()
        val_loss = []
        with torch.no_grad():
            for data in val_data:
                observation = data["observation"]
                if observation.size()[-1] != 0:
                    output = self.model(data["observation"])
                    val_loss.append(self.loss_criterion(output, data["actions"]))
                else:
                    return np.array(val_loss).mean()

        return np.array(val_loss).mean()

    def __call__(self, train_data, val_data):
        for epoch, _ in zip(range(self.n_epochs), tqdm.tqdm(range(self.n_epochs))):
            self.train(epoch=epoch, train_data=train_data, val_data=val_data)

        print(self.loss_history)
        history = pd.DataFrame(self.loss_history)
        history.to_csv("torch_lstm_history.csv")


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from observation_program import ObservationProgram

    allowed_variables = [
        "seeing",
        "clouds",
        "lst",
        "az",
        "alt",
        "zd",
        "ha",
        "airmass",
        "sun_ra",
        "sun_decl",
        "sun_az",
        "sun_alt",
        "sun_zd",
        "sun_ha",
        "moon_ra",
        "moon_decl",
        "moon_az",
        "moon_alt",
        "moon_zd",
        "moon_ha",
        "moon_airmass",
        "moon_phase",
        "moon_illu",
        "moon_Vmag",
        "moon_angle",
        "sky_mag",
        "fwhm",
        "teff",
        "exposure_time",
        "slew",
    ]

    default_obsprog = "../train_configs/default_obsprog.conf"
    obsprog_train = ObservationProgram(config_path=default_obsprog, duration=1)
    datagen_train = ObservationGenerator(
        obsprog_train, included_variables=allowed_variables, n_observation_chains=6
    )
    obsprog_val = ObservationProgram(config_path=default_obsprog, duration=5)
    datagen_val = ObservationGenerator(
        obsprog_val, included_variables=allowed_variables, n_observation_chains=3
    )

    optimizer = torch.optim.SGD
    loss_criterion = torch.nn.MSELoss()

    TrainOfflineLstm(
        input_dim=len(allowed_variables) + 1,
        optimizer=optimizer,
        criterion=loss_criterion,
        n_epochs=50,
    )(datagen_train, datagen_val)
