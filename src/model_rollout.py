"""
Takes model weights that produce a schedule
and produces a schedule at a specific time for it.

Makes the assumption that the schedule is producing weights for an equation;
Producing an equation for each step.
"""
import os

import pandas as pd
import ray.rllib.agents.es as es
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from plots import Plotting

class ModelRollout:
    def __init__(self, experiment_path, scheduler, environment, env_config):
        self.experiment_path = experiment_path

        self.scheduler = scheduler
        self.environment = environment
        self.env_config = env_config

        self.step_env = environment(env_config)

        self.schedule = self.get_schedule()
        self.actions = self.get_actions()

    def get_schedule(self):
        return self.scheduler.schedule

    def get_actions(self):
        return self.scheduler.actions

    def get_model_path(self, checkpoint):
        checkpoint_path = f"{self.experiment_path}/checkpoints"

        history_path = f"{self.experiment_path}/history.csv"

        checkpoints = os.listdir(checkpoint_path)
        checkpoints.sort()
        if checkpoint == 'latest':
            last_checkpoint = checkpoints[-1]
            name = f"checkpoint-{last_checkpoint.split('0')[-1]}"
        else:
            last_checkpoint = f"checkpoint_{str(checkpoint).zfill(6)}"
            name = f"checkpoint-{checkpoint}"

        checkpoint_path = f"{checkpoint_path}/{last_checkpoint}/{name}"
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist!"
        assert os.path.exists(history_path), f"{history_path} does not exist!"

        return checkpoint_path, history_path

    def plot_history(self, history_path):
        history = pd.read_csv(history_path)

        training_hist = pd.DataFrame([json.loads(js) for js in history['info'].str.replace("'",
                                                                                          '"')])
        steps = training_hist['episodes_so_far']
        mean_reward = history["episode_reward_mean"]
        update_ratio = training_hist['update_ratio']
        grad_norm = training_hist['grad_norm']
        weight_norm = training_hist['weights_norm']

        metrics = [mean_reward, update_ratio, grad_norm, weight_norm]
        names = ["Mean Reward", "Weight Update Ratio", "Gradient Norm", "Weight Norm"]
        for metric, name in zip(metrics,names):
            plt.cla()
            plt.scatter(steps, metric)
            plt.title(name)
            plt.xlabel("steps")
            plt.ylabel(name)
            plt.savefig(f"{self.experiment_path}/{name.lower().replace(' ', '_')}_per_step.png")

    def load_model(self, checkpoint):
        model_path, history_path = self.get_model_path(checkpoint)
        self.plot_history(history_path)

        def load_agent():
            agent_config = es.DEFAULT_CONFIG.copy()
            agent_config["env_config"] = self.env_config
            agent_config['num_workers'] = 18
            agent_config['episodes_per_batch'] = 10
            agent_config["evaluation_duration"] = 10
            agent_config['recreate_failed_workers'] = False
            agent_config["model"] = {
                'fcnet_hiddens': []
            }
            agent = es.ESTrainer(config=agent_config, env=self.environment)


            return agent

        agent = load_agent()
        agent.restore(model_path)
        return agent

    def step_model(self, agent, previous_state):
        action_weights = agent.compute_single_action(previous_state)

        try:
            self.scheduler.update(action_weights)
            state = self.scheduler.obsprog.state
            done = self.scheduler.check_endtime(self.scheduler.obsprog.obs)

        except (TypeError, NotImplementedError):
            state, _, done, _ = self.step_env.step(action_weights)

        return state, done

    def wrap_state(self, state):
        wrapped_state = {}
        for key in state:
            wrapped_state[key] = np.array([state[key]], dtype=np.float32)
        return wrapped_state

    def generate_schedule(self, start_date, end_date, save=False, checkpoint='latest'):
        start_time = Time(start_date, format='isot').mjd
        end_time = Time(end_date, format='isot').mjd

        self.step_env.reset()

        self.step_env.mjd = start_time
        self.step_env.start_time, self.step_env.end_time = start_time, end_time

        self.step_env.obs = self.step_env.scheduler.obsprog.observation()
        self.step_env.state = self.step_env.state()

        state = self.step_env.state
        agent = self.load_model(checkpoint)
        done = False

        while not done:
            #state = self.wrap_state(state)
            state, done = self.step_model(agent, state)

        if save:
            if len(self.scheduler.schedule)==0:
                self.scheduler.schedule = self.step_env.scheduler.schedule
            self.scheduler.save(self.experiment_path)

        return self.step_env.scheduler.schedule

    def __call__(self, start_date, end_date, save=True):
        schedule = self.generate_schedule(start_date, end_date, save)
        plotter = Plotting(schedule, self.scheduler.obsprog, len(self.scheduler.actions))
        plotter(self.experiment_path)


if __name__ == "__main__":

    #from rl_scheduler import RLScheduler, RLEnv
    from pure_rl_scheduler import RLEnv, Scheduler

    args = argparse.ArgumentParser()
    args.add_argument("-e", "--experiment_path", default=os.path.abspath("../results/test_dir/"))
    args.add_argument("--scheduler_config_path", default=os.path.abspath("./train_configs"
                                            "/default_schedule.conf"))
    args.add_argument("--obs_config_path", default=os.path.abspath("./train_configs"
                                      "/default_obsprog.conf"))

    args.add_argument("--start_date", default="2021-09-08T01:00:00Z")
    args.add_argument("--end_date", default="2021-09-10T01:00:00Z")

    a = args.parse_args()

    scheduler = Scheduler(a.scheduler_config_path, a.obs_config_path)
    rollout = ModelRollout(experiment_path=a.experiment_path,
                                              scheduler=scheduler,
                                              environment=RLEnv,
                                              env_config={
                                                  "scheduler_config": a.scheduler_config_path,
                                                  "obsprog_config": a.obs_config_path})

    rollout.generate_schedule(a.start_date, a.end_date, save=True)

