"""
this is a single stage RL Scheduler. The policy network is not a network, but rather a
parametric equation, dependent on slew, hour angle, airmass, and moon angle.

The training process is tuning the weights used in the equation to select actions.
This weights will be constant across the whole schedule.

There is no deep network here, just an evolutionary method to update the parametric weights.

"""

import scheduler
import gym
import ray

import numpy as np
import pandas as pd
import ast
from functools import cached_property

from astropy.time import Time
import astropy.units as u


class RLSingleStageSchedule(scheduler.Scheduler):
    def __init__(self, config, obsprog_config):
        super().__init__(config, obsprog_config)

        if self.config.has_option("schedule", "weights"):
            self.initial_weights = ast.literal_eval(self.config.get("schedule", "weights"))
        else:
            self.initial_weights ={"slew": 1, "ha": 1, "airmass": 1, "moon_angle": 1}

        if self.config.has_option("schedule", "powers"):
            self.powers = ast.literal_eval(self.config.get("schedule", "powers"))
        else:
            self.powers = {"slew": 1, "ha": 1, "airmass": 1, "moon_angle": 1}

        self.actions.drop(["band"], axis=1, inplace=True)

    def quality(self):
        # Sum of teff
        # Evaluating on a full generated schedule basis, not a step basis
        return self.schedule.teff.sum()

    def update(self, weights):
        self.obsprog.reset()

        done = False
        while not done:
            action = self.calculate_action(wieghts=weights, observation=self.obsprog.state)
            action["band"] = "g"
            self.feed_action(action)

            reward = action['reward']

            action['mjd'] = self.obsprog.mjd
            self.update_schedule(action, reward)
            done = self.check_endtime(action)

        if "band" in self.schedule.columns:
            self.schedule.drop(["band"], axis=1, inplace=True)

    def single_action_quality(self, weights, observation):

        slew = weights["weight"]*observation["slew"]
        ha = weights["weigh"]*observation['new_obs']["ha"]
        airmass = weights["weight"]*observation["airmass"]
        moon = weights["weight_moon"]*observation["moon_angle"]

        obs_quality = self.initial_weights["slew"]*slew**self.powers["slew"] \
                      + self.initial_weights["ha"]*ha**self.powers["ha"] \
                      + self.initial_weights["airmass"]*airmass**self.powers["airmass"] \
                      + self.initial_weights["moon_angle"]*moon**self.powers["moon_angle"]

        return obs_quality

    def calculate_action(self, **action_params):
        weights = action_params["weights"]
        obs = action_params['observation']

        allowed_actions = self.actions
        quality = []
        allowed = []
        for action in allowed_actions:
            quality.append(self.single_action_quality(weights, obs))
            allowed.append(self.invalid_action(action))

        actions = self.actions.copy()
        actions['reward'] = pd.Series(quality, dtype=float).fillna(0)
        valid_actions = actions[pd.Series(allowed)]

        if len(valid_actions) == 0:
            action = self.obsprog.obs
            action['reward'] = self.invalid_reward

        else:
            action = valid_actions[
                valid_actions['reward'] == valid_actions['reward'].max()
                ].to_dict("records")[0]

        action['mjd'] = self.obsprog.mjd
        if "reward" not in action.keys():
            action['reward'] = self.reward(
                self.obsprog.calculate_exposures(action)
            )

        if "mjd" in action.keys():
            action.pop("mjd")
        return action


class RLSingleStageEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.scheduler = RLSingleStageSchedule(
            config = config["scheduler_config"],
            obsprog_config = config["obsprog_config"]
        )
        self.example_schedule = self.schedule_default()

    def schedule_default(self):
        weights = {weight: 0 for weight in self.action_space.keys()}
        self.scheduler.update(weights)
        return self.scheduler.schedule

    @cached_property
    def action_space(self):
        space = {
            "weight_slew":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "weight_ha":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "weight_airmass":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "weight_moon_angle":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32)
        }
        action_space = gym.spaces.Dict(space)
        return action_space

    @cached_property
    def observation_space(self):

        len_schedule = np.array(self.example_schedule).shape[1]

        space = {}
        space["ra"] = gym.spaces.Box(
                low=self.scheduler.actions.ra.min()-1,
                high=self.scheduler.actions.ra.max()+1,
                shape=(1,len_schedule),
                dtype=np.float32
            )

        space["decl"] = gym.spaces.Box(
                low=self.scheduler.actions.decl.min()-1,
                high=self.scheduler.actions.decl.max()+1,
                shape=(1,len_schedule),
                dtype=np.float32
            )

        space["exposure_time"] = gym.spaces.Box(
                low=self.scheduler.actions.exposure_time.min()-1,
                high=self.scheduler.actions.exposure_time.max()+1,
                shape=(1,len_schedule),
                dtype=np.float32
            )

        space['mjd'] = gym.spaces.Box(
                low=55165, high=70000, shape=(1,len_schedule), dtype=np.float32
            )

        obs_space = gym.spaces.Dict(space)
        return obs_space

    def step(self, action):
        self.scheduler.update(action)
        new_schedule = self.scheduler.schedule
        reward = self.reward(new_schedule)
        done = True
        info = {}
        return new_schedule, reward, done, info

    def reset(self):
        self.scheduler.obsprog.reset()
        return self.example_schedule


class ParametricModel(ray.rllib.models.tf.tf_modelv2.TFModelV2):
    def __init__(self, scheduler, obs_space, action_space, model_config, name):

        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=len(obs_space),
            model_config=model_config,
            name=name
        )

        self.scheduler = scheduler

    def forward(self, input_dict):
        self.scheduler.update(input_dict)
        schedule = np.array(self.scheduler.schedule)
        return schedule