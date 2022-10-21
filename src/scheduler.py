'''
Uses the weights as generated by the rl agent as coefficients to produce an observation for a ground telescope.
Configures each trained model and outputs both the weight producing model and the
'''
import math

import pandas as pd
import numpy as np
import configparser
import os
import ast

import astropy.units as u
from observation_program import ObservationProgram


class Scheduler:
    def __init__(self, config, obsprog_config):
        assert os.path.exists(config)
        self.config = configparser.ConfigParser()
        self.config.read(config)
        duration = self.config.getfloat('schedule', 'length')

        assert os.path.exists(obsprog_config)
        self.obsprog = ObservationProgram(obsprog_config, duration)

        self.actions = self.generate_action_table()
        self.invalid_reward = self.config.getfloat("reward", "invalid_reward")

        schedule_cols = ["mjd", "ra", "decl", "band", "exposure_time", "reward"]
        self.schedule = pd.DataFrame(columns=schedule_cols)

    def generate_action_table(self):
        # Based on the config params
        actions = pd.DataFrame(columns=["ra", 'decl', 'band'])

        min_ra = self.config.getfloat("actions", "min_ra")
        max_ra = self.config.getfloat("actions", "max_ra")
        n_ra = self.config.getint("actions", "num_ra_steps")
        ra_range = np.linspace(min_ra, max_ra, num=n_ra)

        min_decl = self.config.getfloat("actions", "min_decl")
        max_decl = self.config.getfloat("actions", "max_decl")
        n_decl = self.config.getint("actions", "num_decl_steps")
        decl_range = np.linspace(min_decl, max_decl, num=n_decl)

        bands = ast.literal_eval(
                    self.config.get('actions', 'bands'))

        ra_range = ra_range if len(ra_range) != 0 else [0]
        decl_range = decl_range if len(decl_range) != 0 else [0]
        bands = bands if len(bands) != 0 else ['g']

        for ra in ra_range:
            for decl in decl_range:
                for band in bands:
                    new_action = {"ra": ra, "decl": decl, "band": band}
                    new_action = pd.DataFrame(new_action, index=[len(actions)])
                    actions = pd.concat([actions, new_action])

        actions["exposure_time"] = self.config.getfloat("actions",
                                                        "exposure_time")
        return actions

    def update(self, start_date, end_date):
        raise NotImplementedError

    def feed_action(self, action):
        self.obsprog.update_observation(**action)
        new_observation = self.obsprog.state
        return new_observation

    def save(self, outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        schedule_name = f"{outpath.rstrip('/')}/schedule.csv"
        self.schedule.to_csv(schedule_name)

    def reward(self, observation):
        if not self.invalid_action(observation):
            reward = self.invalid_reward
        else:
            reward = self.teff_reward(observation)
        return reward

    def teff_reward(self, observation):
        return observation['teff'] if observation['teff'] is not None else \
            self.invalid_reward

    @staticmethod
    def to_rad(degrees):
        return math.radians(degrees)

    def invalid_action(self, observation):
        RAD = u.rad

        airmass_limit = self.config.getfloat("constraints", "airmass_limit")
        cos_zd_limit = 1.0011 / airmass_limit - 0.0011 * airmass_limit

        # From the spherical cosine formula
        site_lat = self.config.getfloat('site', "latitude")
        cos_lat = np.cos(Scheduler.to_rad(site_lat))
        sin_lat = np.sin(Scheduler.to_rad(site_lat))
        cos_dec = np.cos(observation['decl'] * RAD)
        sin_dec = np.sin(observation['decl'] * RAD)
        cos_ha_limit = (cos_zd_limit - sin_dec * sin_lat) / (cos_dec * cos_lat)

        max_sun_alt = self.config.getfloat("constraints", "max_sun_alt")
        cos_sun_zd_limit = np.cos((90 - max_sun_alt) * RAD)
        cos_sun_dec = np.cos(observation['sun_decl'] * RAD)
        sin_sun_dec = np.sin(observation['sun_decl'] * RAD)
        cos_sun_ha_limit = (cos_sun_zd_limit - sin_sun_dec * sin_lat) / (
                    cos_sun_dec * cos_lat)

        mjd = observation['mjd']


        # Airmass limits
        ha_change = 2 * np.pi * (mjd - observation['mjd']) * 24 / 23.9344696
        ha_change = ha_change * RAD
        ha = observation['ha'] * RAD + ha_change
        in_airmass_limit = np.cos(ha) > cos_ha_limit

        # Moon angle
        min_moon_angle = self.config.getfloat("constraints", "min_moon_angle")
        in_moon_limit = observation['moon_angle'] > min_moon_angle

        # Solar ZD.
        # Ignore Sun's motion relative to ICRS during the exposure
        sun_ha = observation['sun_ha'] * RAD + ha_change
        in_sun_limit = np.cos(sun_ha) < cos_sun_ha_limit

        invalid = in_airmass_limit | in_sun_limit | in_moon_limit
        return invalid

    def calculate_action(self, **action_params):
        raise NotImplementedError

    def update_schedule(self, action, reward):
        action["reward"] = reward
        new_action = pd.DataFrame(action, index=[len(self.schedule)])
        self.schedule = pd.concat([self.schedule, new_action])

    def check_endtime(self, action):
        done = False
        length = self.config.getfloat("schedule", "length")
        end_time = self.obsprog.start_time + length/24
        if action["mjd"]>=end_time:
            done = True
        return done