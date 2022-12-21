import sys

sys.path.append("..")
from observation_program import ObservationProgram
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class ObservationGather:
    def __init__(self, n_observation_chains, chain_duration_days, config_path):
        self.n_observation_chains = n_observation_chains

        self.observation_program = ObservationProgram(
            config_path=config_path, duration=chain_duration_days
        )
        self.observations = pd.DataFrame()
        self.observation_chain = pd.DataFrame()

        self.actions = (5, -5)

    def dataset_visualization(self):
        pass

    def collect_observation_pair(self, ra, decl):
        current_observation = pd.DataFrame(self.observation_program.state, index=[0])
        current_observation["mjd"] = self.observation_program.mjd
        self.observation_program.update_observation(
            ra, decl, band="g", exposure_time=300
        )

        new_observation = pd.DataFrame(self.observation_program.state, index=[0])
        new_observation["mjd"] = self.observation_program.mjd

        current_observation.columns = [f"{col}_t0" for col in current_observation]
        new_observation.columns = [f"{col}_t1" for col in new_observation]
        observation_pair = pd.concat([current_observation, new_observation], axis=1)
        observation_pair["ra"] = ra
        observation_pair["decl"] = decl
        self.observation_chain = pd.concat([self.observation_chain, observation_pair])

    def end_observation_chain(self):
        self.observations = pd.concat([self.observations, self.observation_chain])

        self.observation_chain = pd.DataFrame()
        self.observation_program.reset()

    def select_action(self):
        ra = self.observation_program.ra
        decl = self.observation_program.decl
        ra += self.actions[np.random.randint(0, 2)]
        decl += self.actions[np.random.randint(0, 2)]
        if abs(ra) > 360:
            ra = abs(ra) - 360
        if abs(decl) > 90:
            decl = abs(decl) - 90

        # TODO validity check
        return ra, decl

    def __call__(self, out_path):
        for _ in tqdm(range(self.n_observation_chains)):
            while self.observation_program.mjd < self.observation_program.end_time:
                ra, decl = self.select_action()
                self.collect_observation_pair(ra, decl)
            self.end_observation_chain()

        self.observations.to_csv(out_path)


if __name__ == "__main__":
    obs_config_path = os.path.abspath("../train_configs/default_obsprog.conf")
    ObservationGather(500, 4, obs_config_path)("offline_observations.csv")
