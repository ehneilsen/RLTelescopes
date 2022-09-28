"""
Combines all training pieces and trains and outputs a model
"""

import argparse
import os

from rl_agent import RLAgent
from observation_program import ObservationProgram
from scheduler import Scheduler



def arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--obprog_config", type=str)
    args.add_argument("--rlagent_config", type=str)
    args.add_argument("--scheduler_config", type=str)
    args.add_argument("-i", "--iterations", type=int, default=80)
    args.add_argument("-o", "--output_path", type=str, default="")
    args.add_argument("-c", "--checkpoint", type=int, default=10)

    return args.parse_args()


if __name__ == "__main__":
    args = arguments()

    observations = ObservationProgram(args.obprog_config)
    rl_agent = RLAgent(args.rlagent_config)
    schedule = Scheduler(args.scheduler_config)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i in range(args.iterations):
        schedule.update(rl_agent, observations)

        if i-1 % args.checkpoint==0:
            schedule.save(args.output_path)
            rl_agent.save(args.output_path)


