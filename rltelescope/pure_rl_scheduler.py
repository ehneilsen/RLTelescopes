"""
Implementation of pure rl to directly mimic the traditional methods
Goal: Directly replicate the variable schedulers to prove RL can complete
with tuned analytic methods.

Utilized the adaption of astrotact courtesy of Eric Nielson

If not! Oh well!

Falls into two categories. Temperal weights and full schedule weights.
Either steps through always calculating the next move,
or is given a starting point and calculates the whole schedule until done

Reward is consistently T_eff

"""
import pandas as pd

from scheduler import Scheduler
import gym
from functools import cached_property
import numpy as np
import ast


def reward_function(state):
    # Standard teff reward
    return state["teff"]


def delve_reward(state):

    slew_cubed = state["slew"] ** 3
    hour_angle = state["ha"]
    airmass_cubed = 100 * (state["airmass"] - 1) ** 3

    quality = slew_cubed + hour_angle + airmass_cubed

    return -1 * quality


class RLEnv(gym.Env):
    """
     Just the driver for the unique scheduler designed here
    Initializes the passed scheduler and uses that as an env driver
    """

    def __init__(self, config):
        super().__init__()

        self.scheduler = Scheduler(
            config=config["scheduler_config"], obsprog_config=config["obsprog_config"]
        )
        self.current_reward = self.reward()
        self.obs_vars = self.scheduler.obsprog.state.keys()

        if self.scheduler.config.has_option("schedule", "obs_vars"):
            self.obs_vars = ast.literal_eval(
                self.scheduler.config.get("schedule", "obs_vars")
            )

        self.input_weights = {obs_var: 1 for obs_var in self.obs_vars}
        self.input_powers = {obs_var: 1 for obs_var in self.obs_vars}

        if self.scheduler.config.has_option("schedule", "input_weights"):
            self.input_weights = ast.literal_eval(
                self.scheduler.config.get("schedule", "input_weights")
            )

        if self.scheduler.config.has_option("schedule", "input_powers"):
            self.input_weights = ast.literal_eval(
                self.scheduler.config.get("schedule", "input_powers")
            )

    def reward(self):
        return reward_function(self.scheduler.obsprog.state)

    @cached_property
    def action_space(self):
        return gym.spaces.Discrete(len(self.scheduler.actions))

    @cached_property
    def observation_space(self):
        space = {
            obs_var: gym.spaces.Box(
                low=-10000, high=10000, shape=(1,), dtype=np.float32
            )
            for obs_var in self.obs_vars
        }

        if "mjd" in self.obs_vars:
            space["mjd"] = gym.spaces.Box(
                low=55165, high=70000, shape=(1,), dtype=np.float32
            )

        obs_space = gym.spaces.Dict(space)
        return obs_space

    def state(self):
        full_observation = self.scheduler.obsprog.state
        observation = {
            obs_var: np.array([full_observation[obs_var]], dtype=np.float32)
            for obs_var in self.obs_vars
        }
        return observation

    def update_schedule(self, action, reward):
        action["reward"] = reward
        action = pd.DataFrame(action, index=[0])
        self.scheduler.schedule = pd.concat([self.scheduler.schedule, action])

    def step(self, action):
        action = self.scheduler.actions.iloc[action].to_dict()
        self.scheduler.feed_action(action)
        full_observation = self.scheduler.obsprog.state
        observation = {
            obs_var: np.array(
                [
                    self.input_weights[obs_var]
                    * full_observation[obs_var]
                    * self.input_weights[obs_var]
                ],
                dtype=np.float32,
            )
            for obs_var in self.obs_vars
        }
        reward = self.reward()

        action["mjd"] = self.scheduler.obsprog.mjd
        done = self.scheduler.check_endtime(action)
        info = {}

        self.update_schedule(action, reward)

        return observation, reward, done, info

    def reset(self):
        self.scheduler.obsprog.reset()

        current_obs = self.scheduler.obsprog.state
        observation = {
            obs_var: np.array(
                [
                    self.input_weights[obs_var]
                    * current_obs[obs_var]
                    * self.input_weights[obs_var]
                ],
                dtype=np.float32,
            )
            for obs_var in self.obs_vars
        }

        return observation
