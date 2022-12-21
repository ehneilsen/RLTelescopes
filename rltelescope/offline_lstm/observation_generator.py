"""
A class that generates observations on the fly
according to the params of the observation program it's fed.

Interphases with torch.
"""

import sys

sys.path.append("..")

from observation_program import ObservationProgram
import pandas as pd
import numpy as np
import random
import math

from torch.utils.data import Dataset, DataLoader
import torch


class ObservationGenerator(Dataset):
    def __init__(
        self,
        observation_program: ObservationProgram,
        n_observation_chains: int = 20,
        batch_size: int = 640,
        bands: list = None,
        n_observation_sites: int = 50,
        ra_range: tuple = None,
        delc_range: tuple = None,
        included_variables=None,
    ):

        self.observation_program = observation_program
        self.n_observation_chains = n_observation_chains
        self.batch_size = batch_size

        default_vars = pd.DataFrame(observation_program.state, index=[0]).columns
        self.included_variables = (
            included_variables if included_variables is not None else default_vars
        )

        self.step_actions = (1, 0, -1)
        self.bands = ["g"] if bands is None else bands
        self.n_observation_sites = n_observation_sites
        self.set_action_vars(ra_range, delc_range)

        self.actions = pd.concat(
            [self.generate_action_set() for _ in range(n_observation_chains)]
        )

    def set_action_vars(self, ra_range, delc_range):
        ra_range = (0, 361) if ra_range is None else ra_range
        delc_range = (-90, 91) if delc_range is None else delc_range

        ra_step = np.abs(ra_range[0] - ra_range[-1]) / self.n_observation_sites
        delc_step = np.abs(delc_range[0] - delc_range[-1]) / self.n_observation_sites

        self.ra_arrangement = np.arange(ra_range[0], ra_range[-1], ra_step)
        self.delc_arrangement = np.arange(delc_range[0], delc_range[-1], delc_step)

    def generate_action_set(self):

        actions = pd.DataFrame()
        ra_index = random.randint(0, len(self.ra_arrangement) - 1)
        delc_index = random.randint(0, len(self.delc_arrangement) - 1)

        sites = [[self.ra_arrangement[ra_index], self.delc_arrangement[delc_index]]]

        for _ in range(self.n_observation_sites):
            ra_index += self.step_actions[random.randint(0, len(self.step_actions) - 1)]
            delc_index += self.step_actions[
                random.randint(0, len(self.step_actions) - 1)
            ]

            ra_index = ra_index if ra_index < len(self.ra_arrangement) else 0
            delc_index = delc_index if delc_index < len(self.delc_arrangement) else 0

            new_ra = self.ra_arrangement[ra_index]
            new_delc = self.delc_arrangement[delc_index]

            sites.append([new_ra, new_delc])
        sites = np.array(sites)
        for band in self.bands:
            actions = pd.concat(
                [
                    actions,
                    pd.DataFrame(
                        {
                            "ra": sites[:, 0].astype(np.float32),
                            "delc": sites[:, 1].astype(np.float32),
                            "band": band,
                        }
                    ),
                ]
            )
        return actions

    def __getitem__(self, item):
        batch_lower_index = item * self.batch_size
        batch_higher_index = (item + 1) * self.batch_size
        batch_actions = self.actions.iloc[batch_lower_index:batch_higher_index]

        batch = self.data_generation(batch_actions.to_dict(orient="records")).astype(
            np.float32
        )
        actions = batch_actions[["ra", "delc"]].values.astype(np.float32)

        return {
            "observation": torch.from_numpy(batch),
            "actions": torch.from_numpy(actions),
        }

    def __len__(self):
        return math.ceil(len(self.actions) / self.batch_size)

    def data_generation(self, actions):
        observation_chain = pd.DataFrame()
        for action in actions:
            ra = action["ra"]
            delc = action["delc"]
            band = action["band"]

            current_observation = pd.DataFrame(
                self.observation_program.state, index=[0]
            )
            current_observation = current_observation[self.included_variables]
            current_observation["mjd"] = self.observation_program.mjd

            self.observation_program.update_observation(
                ra, delc, band, exposure_time=300
            )

            new_observation = pd.DataFrame(self.observation_program.state, index=[0])
            new_observation = new_observation[self.included_variables]
            new_observation["mjd"] = self.observation_program.mjd

            current_observation.columns = [f"{col}_t0" for col in current_observation]
            new_observation.columns = [f"{col}_t1" for col in new_observation]

            observation_pair = pd.concat([current_observation, new_observation], axis=1)
            observation_chain = pd.concat([observation_chain, observation_pair])

        observation_chain.reset_index(drop=True, inplace=True)

        observation_chain = np.stack(
            [
                observation_chain[
                    [col for col in observation_chain.columns if "t0" in col]
                ].values,
                observation_chain[
                    [col for col in observation_chain.columns if "t1" in col]
                ].values,
            ]
        )

        return observation_chain


if __name__ == "__main__":
    default_obsprog = "../train_configs/default_obsprog.conf"
    obsprog = ObservationProgram(config_path=default_obsprog, duration=1)
    datagen = ObservationGenerator(obsprog)
    datagen.__getitem__(0)
