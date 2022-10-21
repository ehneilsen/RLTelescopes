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
        self.schedule=schedule
        self.obsprog=obsprog
        self.n_sites = n_sites

    def schedule_obs(self):
        results = {
            "reward":[],
            "teff":[]
        }
        schedule = pd.DataFrame(self.schedule).to_dict('index')
        for step in schedule:
            step = schedule[step]
            obs = self.obsprog.calculate_exposures(step)
            results["reward"].append(obs['reward'])
            results['teff'].append(obs['teff'])

        return results

    def schedule_metrics(self):
        rewards = self.schedule_obs()
        mean_teff = pd.Series(rewards['teff']).mean()
        max_teff = pd.Series(rewards['teff']).max()
        std_teff = pd.Series(rewards['std']).max()
        total_reward = pd.Series(rewards['reward']).sum()

        grouping = self.schedule.groupby(["ra", 'decl', 'band']).index
        coverage = len(grouping.unique())/self.n_sites
        uniformity = grouping.std()

        return {
            "Max Teff": max_teff,
            "Mean Teff": mean_teff,
            "Std Teff": std_teff,
            "Total Reward": total_reward,
            "Coverage": coverage,
            "Uniformity": uniformity
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

        # band distibution
        plt.cla()
        plt.clf()
        bands_df = pd.DataFrame({
            "bands": band,
            metric_name: metric
        })

        band_sum = bands_df.groupby("bands").sum()
        plt.hist([i for i in range(len(band_sum))], band_sum[metric_name].values)
        plt.xlabel("bands")
        plt.ylabel(f"Sum {metric_name}")
        plt.title("Band Distribution")
        plt.savefig(f"{save_path}/band_distribution_{metric_name.lower()}.png")

    def __call__(self, save_path):
        rewards = self.schedule_obs()

        # Metrics
        reward, teff = rewards['reward'], rewards['teff']

        mjd = self.schedule['mjd']
        ra = self.schedule['ra']
        decl = self.schedule['decl']
        band = self.schedule['band']

        self.plot_metric_progress(mjd, ra, decl, band, reward, "Reward", save_path)
        self.plot_metric_progress(mjd, ra, decl, band, teff, "T_eff", save_path)


if __name__ == "__main__":

    from observation_program import ObservationProgram


    args = argparse.ArgumentParser()
    args.add_argument("--schedule_path", type=str, default="../results/test_dir/schedule.csv")
    args.add_argument("--obsprog_config", type=str, default="./train_configs/default_obsprog.conf")
    args.add_argument("--n_sites", type=int, default=10)
    a = args.parse_args()

    schedule = pd.read_csv(os.path.abspath(a.schedule_path))
    obsprog = ObservationProgram(config_path=a.obsprog_config, duration=0)
    out_path = os.path.dirname(a.schedule_path)

    Plotting(schedule, obsprog, a.n_sites)(out_path)
