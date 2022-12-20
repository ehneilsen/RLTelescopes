"""
Simple scheduler that just selects the next event with the lowest slew

"""
import argparse
from scheduler import Scheduler
import pandas as pd
from tqdm import tqdm
import os

from astropy.time import Time


class SqueScheduler(Scheduler):
    def __init__(self, scheduler_config, obsprog_config):
        super().__init__(scheduler_config, obsprog_config)

    def update(self, start_date, end_date=None):

        start_time = Time(start_date, format='isot').mjd
        end_time = Time(end_date, format='isot').mjd

        self.obsprog.reset()

        self.obsprog.mjd = start_time
        self.obsprog.start_time, self.obsprog.end_time = start_time, end_time

        length = self.config.getfloat("schedule", "length")
        n_actions = len(self.actions)
        time_per_action = self.config.getfloat("actions", "exposure_time") +  \
                          self.obsprog.slew_rate*n_actions/360
        n_steps = int(
            (length * 60 * 60) / time_per_action)

        nights = int(length/24)

        done = False
        while not done:
            action = self.calculate_action()
            self.feed_action(action)

            reward = action['reward']
            action['mjd'] = self.obsprog.mjd
            self.update_schedule(action, reward)
            done = self.check_endtime(action)

    @staticmethod
    def quality(new_obs):
        quality = abs(new_obs['airmass'])
        return quality

    def calculate_action(self, **kwargs):
        # Iterate through actions
        valid_actions = []
        quality = []
        actions = self.actions.to_dict("records")
        for action in actions:
            action['mjd'] = self.obsprog.mjd
            new_obs = self.obsprog.calculate_exposures(action)
            quality.append(SqueScheduler.quality(new_obs=new_obs))
            action['reward'] = quality[-1]
            valid_actions.append(self.invalid_action(new_obs))

        actions = pd.DataFrame(actions)
        actions['reward'] = pd.Series(quality, dtype=float).fillna(0)
        valid_actions = actions[pd.Series(valid_actions)]
        non_zero = valid_actions[valid_actions['reward'] != 0]

        if len(non_zero) == 0:
            action = self.obsprog.obs

        else:
            action = non_zero[
                non_zero['reward'] == non_zero['reward'].min()
                ].to_dict("records")[0]

        if "mjd" in action.keys():
            action.pop("mjd")

        return action


if __name__ == "__main__":
    scheduler_config_path = os.path.abspath("train_configs"
                                            "/default_schedule.conf")
    obs_config_path = os.path.abspath("train_configs"
                                      "/default_obsprog.conf")
    out_path = os.path.abspath("../results/low_airmass_default")

    args = argparse.ArgumentParser()
    args.add_argument("--obsprog_config", type=str, default=obs_config_path)
    args.add_argument("--schedule_config", type=str,
                      default=scheduler_config_path)

    args.add_argument("--start_date", default="2021-09-08T01:00:00Z")
    args.add_argument("--end_date", default="2021-09-10T01:00:00Z")

    args.add_argument("-o", "--out_path", type=str, default=out_path)
    a = args.parse_args()

    scheduler = SqueScheduler(a.schedule_config, a.obsprog_config)
    scheduler.update(a.start_date, a.end_date)
    scheduler.save(a.out_path)
