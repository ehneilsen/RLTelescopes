"""
A stealth copy of delve's obztrak
Uses the same variables to select an action
based on the best choices at any given time

"""
import astropy.units as u
import numpy as np

import pandas as pd
from tqdm import tqdm
from src.schedule_trainer.scheduler import Scheduler


class VariableScheduler(Scheduler):
    def __init__(self, config, obsprog):
        super().__init__(config, obsprog)

    def update(self):
        self.obsprog.reset()
        length = self.config.getfloat("schedule", "length")
        n_steps = int((length*60*60)/self.config.getfloat("actions",
                                                       "exposure_time"))+1

        for _, _ in zip(range(n_steps), tqdm(range(n_steps))):
            original_time = str(self.obsprog.mjd)
            action = self.calculate_action()
            self.feed_action(action)

            assert str(self.obsprog.mjd) != original_time

            reward = action['reward']
            action['mjd'] = self.obsprog.mjd
            self.update_schedule(action, reward)

    @staticmethod
    def quality(new_obs):
        slew_cubed = new_obs['slew']**3
        hour_angle = new_obs['ha']
        airmass_cubed = 100*(new_obs['airmass']-1)**3

        quality = slew_cubed + hour_angle + airmass_cubed

        return quality

    def calculate_action(self, **kwargs):
        # Iterate through actions
        valid_actions = []
        quality = []
        actions = self.actions.to_dict("records")
        for action in actions:
            action['mjd'] = self.obsprog.mjd
            new_obs = self.obsprog.calculate_exposures(action)
            quality.append(VariableScheduler.quality(new_obs=new_obs))
            valid_actions.append(self.invalid_action(new_obs))

        actions = self.actions.copy()
        actions['reward'] = pd.Series(quality, dtype=float).fillna(0)
        valid_actions = actions[pd.Series(valid_actions)]

        if len(valid_actions) == 0:
            action = self.obsprog.obs

        else:
            action = valid_actions[
                valid_actions['reward']==valid_actions['reward'].max()
            ].to_dict("records")[0]

        if "mjd" in action.keys():
            action.pop("mjd")

        return action
