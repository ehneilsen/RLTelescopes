"""
A two stage predictor.  An NN is trained on N steps of data.
The output is the weights used to predict the next action (for each weight)

"""

from scheduler import Scheduler
import gym
from functools import cached_property
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from collections import OrderedDict


class RLScheduler(Scheduler):
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

        if self.config.has_option("schedule", "min_max"):
            min_max = self.config.get("schedule", "min_max")
        else:
            min_max = "max"

        assert min_max in ["min", "max"], "Parameter 'schedule min_max' must be set to either " \
                                          "'min' or 'max'"
        self.selector = min_max

    def update(self, nn_weights):
        action = self.calculate_action(action=nn_weights)
        self.feed_action(action)

        reward = action['reward']
        action['mjd'] = self.obsprog.mjd
        for n in nn_weights:
            action[n] = nn_weights[n]
        self.update_schedule(action, reward)

    def quality(self, new_obs, nn_action):
        slew = nn_action["weight_slew"]*new_obs["slew"]
        ha = nn_action["weight_ha"]*new_obs["ha"]
        airmass = nn_action["weight_airmass"]*new_obs["airmass"]
        moon = nn_action["weight_moon_angle"]*new_obs["moon_angle"]

        obs_quality = self.initial_weights["slew"]*slew**self.powers["slew"] \
                      + self.initial_weights["ha"]*ha**self.powers["ha"] \
                      + self.initial_weights["airmass"]*airmass**self.powers["airmass"] \
                      + self.initial_weights["moon_angle"]*moon**self.powers["moon_angle"]

        return obs_quality

    def action_weights(self, nn_action):
        valid_actions = []
        quality = []
        actions = self.actions.to_dict("records")
        for action in actions:
            action['mjd'] = self.obsprog.mjd
            new_obs = self.obsprog.calculate_exposures(action)
            quality.append(self.quality(new_obs=new_obs, nn_action=nn_action))
            valid_actions.append(self.invalid_action(new_obs))

        actions = self.actions.copy()
        actions['reward'] = pd.Series(quality, dtype=float).fillna(0)
        valid_actions = actions[pd.Series(valid_actions)]
        return valid_actions

    def calculate_action(self, **kwargs):
        nn_action = kwargs["action"]
        valid_actions = self.action_weights(nn_action)

        # *YOU*
        if len(valid_actions) == 0:
            action = self.obsprog.obs
            action['reward'] = self.invalid_reward

        else:
            if self.selector == "max":
                action = valid_actions[
                    valid_actions['reward'] == valid_actions['reward'].max()
                ].to_dict("records")[0]
            else:
                action = valid_actions[
                    valid_actions['reward'] == valid_actions['reward'].min()
                    ].to_dict("records")[0]

        action['mjd'] = self.obsprog.mjd
        if "reward" not in action.keys():
            action['reward'] = self.reward(
                self.obsprog.calculate_exposures(action)
            )

        if "mjd" in action.keys():
            action.pop("mjd")
        return action


class RLEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.scheduler = RLScheduler(
            config=config["scheduler_config"],
            obsprog_config=config["obsprog_config"]
        )
        self.current_reward = self.reward()

    def reward(self):
        return self.scheduler.reward(self.scheduler.obsprog.state)

    @cached_property
    def action_space(self):
        space = {
            "weight_slew":gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32),
            "weight_ha":gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32),
            "weight_airmass":gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32),
            "weight_moon_angle":gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32)
        }
        action_space = gym.spaces.Dict(space)
        return action_space

    @cached_property
    def observation_space(self):
        obs = self.scheduler.obsprog.state
        obs_vars = obs.keys()
        space = {
            obs_var:gym.spaces.Box(
                low=-10000, high=10000, shape=(1,), dtype=np.float32
            )
            for obs_var in obs_vars
        }
        space['mjd'] = gym.spaces.Box(
                low=55165, high=70000, shape=(1,), dtype=np.float32
            )
        obs_space = gym.spaces.Dict(space)
        return obs_space

    def step(self, action):
        original_time = self.scheduler.obsprog.mjd

        true_action = self.scheduler.calculate_action(
            action=action,
            obs=self.scheduler.obsprog.obs
        )
        self.scheduler.feed_action(true_action)

        true_action['mjd'] = original_time
        observation = self.scheduler.obsprog.state
        reward = self.reward()
        done = self.scheduler.check_endtime(true_action)
        info = {}

        observation = {
            var: np.array([observation[var]], dtype=np.float32)
            for var in observation.keys()
        }

        return observation, reward, done, info

    def reset(self):
        self.scheduler.obsprog.reset()

        current_obs = self.scheduler.obsprog.state
        observation = {
            obs_var: np.array([val], dtype=np.float32)
                for obs_var, val in current_obs.items()
            }
        return observation
