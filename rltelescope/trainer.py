"""
Combines all training pieces and trains and outputs a model
Uses RLlib's evostrat agent
Everything else is custom. See notes for those pieces indiv'ly
"""

import sys

import argparse
import os

import ray.rllib.algorithms.es as es
from ray.rllib.models import ModelCatalog
import ray
import tqdm
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from pure_rl_scheduler import RLEnv

scheduler_config_path = os.path.abspath(
    "train_configs" "/weighted_delve_schedule_rl.config"
)
obs_config_path = os.path.abspath("train_configs" "/default_obsprog.conf")
out_path = os.path.abspath("../results/weighted_delve_rl")


def arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--obsprog_config", type=str, default=obs_config_path)
    args.add_argument("--schedule_config", type=str, default=scheduler_config_path)
    args.add_argument("-i", "--iterations", type=int, default=80)
    args.add_argument("-o", "--out_path", type=str, default=out_path)

    return args.parse_args()


def make_agent(env, env_config):

    agent_config = es.DEFAULT_CONFIG.copy()
    agent_config["env_config"] = env_config
    agent_config["num_workers"] = 50
    # agent_config['episodes_per_batch'] = 10
    # agent_config["evaluation_duration"] = 10
    agent_config["model"] = {"fcnet_hiddens": []}
    agent_config["framework"] = "torch"
    agent = es.ES(config=agent_config, env=env)
    return agent


if __name__ == "__main__":

    args = arguments()
    ray.init()

    agent = make_agent(
        RLEnv,
        {
            "scheduler_config": args.schedule_config,
            "obsprog_config": args.obsprog_config,
        },
    )

    checkpoints_outpath = f"{args.out_path}/checkpoints/"
    if not os.path.exists(checkpoints_outpath):
        os.makedirs(checkpoints_outpath)

    history = {}
    print(f"Beginning training for {args.iterations} iterations")
    for i in tqdm.trange(args.iterations):
        # Training loop

        step_history = agent.train()
        agent.save(checkpoints_outpath)

        history[i] = step_history
        pd.DataFrame(history).T.to_csv(f"{args.out_path}/history.csv")

    ray.shutdown()
