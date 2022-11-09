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
import tensorflow as tf

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
        return self.schedule.reward.sum()

    def assign_init(self, mjd, init_ra, init_decl, obs_time):
        if mjd is not None:
            self.obsprog.mjd = mjd
        if init_ra is not None:
            self.obsprog.ra = init_ra
        if init_decl is not None:
            self.obsprog.decl = init_decl
        if obs_time is not None:
            self.obsprog.exposure_time = obs_time

    def update(self, weights, mjd=None, init_ra=None, init_decl=None, obs_time=300):
        self.obsprog.reset()
        self.assign_init(mjd, init_ra, init_decl, obs_time)

        done = False
        while not done:
            telescope_action = self.calculate_action(weights=weights, obs=self.obsprog.state)
            telescope_action["band"] = "g"
            self.feed_action(telescope_action)

            reward = telescope_action['reward']

            telescope_action['mjd'] = self.obsprog.mjd
            self.update_schedule(telescope_action, reward)
            done = self.check_endtime(telescope_action)

        if "band" in self.schedule.columns:
            self.schedule.drop(["band"], axis=1, inplace=True)

        return self.schedule

    def single_action_quality(self, weights, observation):

        slew = weights["slew"]*observation["slew"]
        ha = weights["ha"]*observation["ha"]
        airmass = weights["airmass"]*observation["airmass"]
        moon = weights["moon_angle"]*observation["moon_angle"]

        obs_quality = self.initial_weights["slew"]*slew**self.powers["slew"] \
                      + self.initial_weights["ha"]*ha**self.powers["ha"] \
                      + self.initial_weights["airmass"]*airmass**self.powers["airmass"] \
                      + self.initial_weights["moon_angle"]*moon**self.powers["moon_angle"]

        return obs_quality

    def calculate_action(self, weights=None, obs=None):
        allowed_actions = self.actions
        quality = []
        allowed = []
        for action in allowed_actions.to_dict("records"):
            quality.append(self.single_action_quality(weights, obs))
            action['mjd'] = self.obsprog.mjd
            action['band'] = 'g'
            possible_exposure = self.obsprog.calculate_exposures(action)
            allowed.append(self.invalid_action(possible_exposure))

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

        self.scheduler = RLSingleStageSchedule(
            config=config["scheduler_config"],
            obsprog_config=config["obsprog_config"]
        )

        super().__init__()
        self.example_schedule = self.schedule_default()

    def schedule_default(self):
        weights = {weight: 0 for weight in self.action_space.keys()}
        self.scheduler.update(weights=weights)
        return self.scheduler.schedule

    @cached_property
    def action_space(self):
        space = {
            "slew":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "ha":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "airmass":gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32),
            "moon_angle":gym.spaces.Box(
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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )

        self.scheduler = RLSingleStageSchedule(model_config["custom_model_config"][
                                                   "scheduler_config"],
                                               model_config["custom_model_config"][
                                                   "obsprog_config"])
        self.model = self.schedule_model()

    def schedule_model(self):
        input = tf.keras.layers.Input(shape=self.obs_space.shape)
        output = tf.keras.layers.Dense(1, activation="linear")(input)
        model = tf.keras.Model(input, output)
        return model

    @tf.function
    def convert_weights(self, weights):
        return

    def forward(self, input_dict, state, seq_lens):
        session = tf.compat.v1.Session()

        init_schedule_conditions = input_dict["obs"]
        ra = init_schedule_conditions['ra']
        delc = init_schedule_conditions['decl']
        mjd = init_schedule_conditions["mjd"]

        #input_model = self.model(input_dict['obs_flat'])#.layers[0].weights[0]
        with session as sess:

            self.model.compile(loss="mse")
            #output = self.model(input_dict['obs_flat'])
            schedule_weights = self.model.layers[-1].weights[0]
            schedule_weights = {
                key: weight
                for key, weight
                in zip(
                    init_schedule_conditions.keys(),
                    schedule_weights.eval(session=sess))
            }
        print(schedule_weights)
        schedule_update = {
            "weights": schedule_weights, "mjd": mjd, "init_ra": ra,"init_decl": delc
        }
        schedule = self.scheduler.update(**schedule_update)

        return schedule, state

    def value_function(self):
        return self.scheduler.quality()