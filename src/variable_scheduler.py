"""
A stealth copy of delve's obztrak
Uses the same variables to select an action
based on the best choices at any given time

"""
import astropy.units as u

import numpy as np

import pandas as pd
from tqdm import tqdm
from scheduler import Scheduler
import os
import argparse

from astropy.time import Time


class VariableScheduler(Scheduler):
    def __init__(self, config, obsprog):
        super().__init__(config, obsprog)

    def update(self, start_date, end_date):
        self.obsprog.reset()

        start_time = Time(start_date, format='isot').mjd
        end_time = Time(end_date, format='isot').mjd

        self.obsprog.reset()

        self.obsprog.mjd = start_time
        self.obsprog.start_time, self.obsprog.end_time = start_time, end_time

        done = False
        while not done:
            original_time = str(self.obsprog.mjd)
            action = self.calculate_action()
            self.feed_action(action)

            reward = action['reward']
            action['mjd'] = self.obsprog.mjd
            self.update_schedule(action, reward)
            done = self.check_endtime(action)

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
            action['reward'] = self.invalid_reward

        else:
            action = valid_actions[
                valid_actions['reward']==valid_actions['reward'].max()
            ].to_dict("records")[0]

        if "mjd" in action.keys():
            action.pop("mjd")

        return action


if __name__ == "__main__":
    scheduler_config_path = os.path.abspath("train_configs"
                                            "/default_schedule.conf")
    obs_config_path = os.path.abspath("train_configs"
                                      "/default_obsprog.conf")
    out_path = os.path.abspath("../results/variable_default")

    args = argparse.ArgumentParser()
    args.add_argument("--obsprog_config", type=str, default=obs_config_path)
    args.add_argument("--schedule_config", type=str,
                      default=scheduler_config_path)
    args.add_argument("--start_date", default="2021-09-08T01:00:00Z")
    args.add_argument("--end_date", default="2021-09-10T01:00:00Z")

    args.add_argument("-o", "--out_path", type=str, default=out_path)
    a = args.parse_args()

    scheduler = VariableScheduler(a.schedule_config, a.obsprog_config)
    scheduler.update(a.start_date, a.end_date)
    scheduler.save(a.out_path)
