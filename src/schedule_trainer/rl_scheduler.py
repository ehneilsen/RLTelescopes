"""
A two stage predictor.  An NN is trained on N steps of data.
The output is the weights used to predict the next action (for each weight)


"""

from src.schedule_trainer.scheduler import Scheduler
import gym
import configparser
from functools import cached_property
import numpy as np
import pandas as pd
from tqdm import tqdm

class RLScheduler(Scheduler):
    def __init__(self, config, obsprog):
        super().__init__(config, obsprog)

    def update(self, nn_weights):
        ## Rolling out a schedule with given weights
        self.obsprog.reset()
        length = self.config.getfloat("schedule", "length")
        n_steps = int((length * 60 * 60) / self.config.getfloat("actions",
                                                                "exposure_time")) + 1

        for _, _ in zip(range(n_steps), tqdm(range(n_steps))):
            action = self.calculate_action(action=nn_weights)
            self.feed_action(action)

            reward = action['reward']
            action['mjd'] = self.obsprog.mjd
            self.update_schedule(action, reward)

    def quality(self, new_obs, nn_action):
        slew = nn_action["weight_slew"]*new_obs["slew"]
        ha = nn_action["weight_ha"]*new_obs["ha"]
        airmass = nn_action["weight_airmass"]*new_obs["airmass"]
        moon = nn_action["weight_moon_angle"]*new_obs["moon_angle"]
        obs_quality = slew + ha + airmass + moon
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

        if len(valid_actions) == 0:
            action = self.obsprog.obs

        else:
            action = valid_actions[
                valid_actions['reward'] == valid_actions['reward'].max()
            ].to_dict("records")[0]

        action['mjd'] = self.obsprog.mjd
        action['reward'] = self.reward(
            self.obsprog.calculate_exposures(action)
        )

        if "mjd" in action.keys():
            action.pop("mjd")
        return action


class RLEnv(gym.Env):
    def __init__(self, config, scheduler):
        super().__init__()

        self.current_reward = 0
        self.scheduler = scheduler
        self.action_space = gym.spaces.Discrete(len(scheduler.actions) + 1)

    @cached_property
    def observation_space(self):
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
        obs_space = gym.spaces.Dict(space)
        return obs_space

    def step(self, action):

        true_action = self.scheduler.calculate_action(
            action=action,
            obs=self.scheduler.obsprog.obs
        )

        self.scheduler.obsprog.feed_action(true_action)
        observation = self.scheduler.obsprog.state

        reward = self.scheduler.reward()
        done = self.scheduler.check_endtime(action)
        info = ""
        return observation, reward, done, info

    def reset(self):
        self.scheduler.obsprog.reset()

