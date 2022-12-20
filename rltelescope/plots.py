"""
Plots to specifically evaluate different generated schedules, and rl models when applicable
"""

import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import numpy as np


class Plotting:
    def __init__(self, schedule, obsprog, n_sites):
        self.schedule = schedule
        self.obsprog = obsprog
        self.n_sites = n_sites

    def schedule_obs(self):

        schedule = pd.DataFrame(self.schedule).to_dict('index')
        observations = []
        for step in schedule:
            step_vars = schedule[step]
            obs = self.obsprog.calculate_exposures(step_vars)

            for key in obs.keys():
                schedule[step][key] = obs[key]

            observations.append(schedule[step])

        observations = pd.DataFrame(observations)
        return observations

    def schedule_metrics(self, schedule_states):
        mean_teff = pd.Series(schedule_states['teff']).mean()
        max_teff = pd.Series(schedule_states['teff']).max()
        std_teff = pd.Series(schedule_states['teff']).std()
        sum_teff = pd.Series(schedule_states['teff']).sum()
        total_reward = pd.Series(schedule_states['reward']).sum()

        grouping = self.schedule.groupby(["ra", 'decl', 'band'])
        coverage = len(grouping.groups)/self.n_sites

        return {
            "Max Teff": max_teff,
            "Mean Teff": mean_teff,
            "Total Teff": sum_teff,
            "Std Teff": std_teff,
            "Total Reward": total_reward,
            "Coverage": coverage,
        }

    def plot_metric_progress(self, mjd, ra, decl, band, metric, metric_name, save_path):
        plt.cla()
        plt.clf()
        # mjd vs metric
        plt.scatter(mjd, metric)
        plt.ylabel(metric_name)
        plt.xlabel("mjd (days)")
        plt.title(f"Time vs {metric_name}")
        plt.savefig(f"{save_path}/mjd_vs_{metric_name.lower()}.png")

        # position distribution
        plt.cla()
        plt.clf()
        fig = plt.subplot(projection='polar')
        fig.set_theta_zero_location("N")
        fig.set_yticklabels([])
        fig.set_xticklabels([])
        fig.set_title("Position Distribution")
        fig.set_ylim(-180, 180)

        f = fig.scatter([ra * np.pi / 180], [decl * np.pi / 180], c=metric)
        plt.colorbar(f)
        plt.savefig(f"{save_path}/position_{metric_name.lower()}.png")


    def __call__(self, save_path):
        rewards = self.schedule_obs()
        print(self.schedule_metrics(rewards))
        metrics = pd.DataFrame(self.schedule_metrics(rewards), index=[0])

        # Metrics
        reward, teff = rewards['reward'], rewards['teff']

        mjd = self.schedule['mjd']
        ra = self.schedule['ra']
        decl = self.schedule['decl']
        band = self.schedule['band']

        self.plot_metric_progress(mjd, ra, decl, band, reward, "Reward", save_path)
        self.plot_metric_progress(mjd, ra, decl, band, teff, "T_eff", save_path)

        rewards.to_csv(f"{save_path}/schedule_states.csv")
        metrics.to_csv(f"{save_path}/schedule_metrics.csv")



if __name__ == "__main__":

    from observation_program import ObservationProgram

    args = argparse.ArgumentParser()
    args.add_argument("-s","--schedule_path", type=str, default="../results/test_dir/schedule.csv")
    args.add_argument("--obsprog_config", type=str, default="./train_configs/default_obsprog.conf")
    args.add_argument("--n_sites", type=int, default=10)
    a = args.parse_args()

    schedule = pd.read_csv(os.path.abspath(a.schedule_path))
    obsprog = ObservationProgram(config_path=a.obsprog_config, duration=0)
    out_path = os.path.dirname(a.schedule_path)

    Plotting(schedule, obsprog, a.n_sites)(out_path)
