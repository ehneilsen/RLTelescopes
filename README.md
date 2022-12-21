
![status](https://img.shields.io/badge/License-MIT-lightgrey)

## Summary

It's broken. :<


## Installation
#### Install from pip
Simply run

`pip install git+https://github.com/deepskies/RLTelescopes@main`

This will install the project with al its requirements.

#### Install from source
It requires both this repo and the repo [SkyBright](https://github.com/ehneilsen/skybright).
This is included in the pyproject.toml

Due to the limitation of rllib, a conda environment is required if running on Mac with an M1 chip if you wish to use tensorflow/keras.
For this reason, the RL Trainer uses Torch as a backend.

The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command

`poetry install`

Assuming you have poetry installed on your base environment.
This will use lock file to install all the correct versions.
To use the installed environment, use the command `poetry shell` to enter it.
The command `exit` will take you out of this environment as it would for any other type of virtual environment.

Otherwise, you can use the `pyproject.toml` with your installer of choice.


## Quickstart
To immediately start training a model and verify your installation, use the command
```
python3 src/trainer.py
   --obsprog_config src/train_configs/default_obsprog.conf
   --schedule_config src/train_configs/default_schedule.conf
   --iterations 5
   --out_path results/test_schedule
```

To evaluate an already generated schedule, use the command
```
python3 src/plotting.py
    --schedule_path <Path to schedule csv>
    --obsprog_config src/train_configs/default_obsprog.conf
    --n_sites <Number of possible sites for a schedule to have visited>
```

Or generate a schedule using a reinforcement learning model (verify the agent is correctly
configured beforehand.)
```
python3 src/model_rollout.py
    --experiment_path <Directory where the trained weights are stored, the program will
    automatically pick the latest weight>
    --scheduler_config_path src/train_configs/default_schedule.conf
    --obs_config_path src/train_configs/default_obsprog.conf
    --start_date <Date in YYYY-MM-DDTHH:MM:SSZ>
    --end_date <Date in YYYY-MM-DDTHH:MM:SSZ>
```

## Documentation

All inputs for each file individually can be found by passing -h as a parameter to any file,
such as `python3 file.py -h`.


This project is split into 3 major parts.

#### The Observation Program

The observation program is a single file (observation_program.py) which simulates a period of
time during which the schedule is executed. It calculates parameters of the night sky as a
function fo time and position. A single observation is considered a combination of sky
coordinates (right accession, declination), time (in mean julian date), and observation filter
(referred to as band).

To start an instance of the observer, a configuration file is needed. An example can be seen in
src/train_configs/default_obsprog.config, but briefly, it requires the position of the ground
observatory, specifics of the types of observation it can make, and parameters for the sky
simulation required by `skybright`.

#### The Scheduler

The scheduler (scheduler.py) is an semi-abstract class containing the methods to automate the
generation of sky surveys from a ground position.

It's main function is that of a driver and recorder for the observation program. It initializes
an instance of the observation program to use, and steps through it, selecting the next site to
visit based on the 'select_action' function, which is left non-implemented in the abstract class.

It has a number of children classes:
* _Low Airmass Scheduler_ - Selects the next action based pured on what action next in time
  sequence has the lowest airmass and is not below the horizon.
* _Variable Schedule_ - Selects the next site based on which site at the next observation time
  the equation `R = min(slew^3+ ha + 100*(airmass-1)^3`
* Sequential Schedule_ - Selects the next site that is the shorted slew distance away.
* _RL Schedule_ - Trains an RL Model that selects the weights for a selection equation dependent
  on time, such that the selected site is picked via the equation
 `R=min(NN_slew*slew + NN_ha*ha + NN_airmass*airmass + NN_moon angle*moon angle)`
* _Pure RL Schedule_ - Selects a site based on the selection of an RL trained network. The inputs
  and specifics of this network can be decided in the schedule parameters file.

Any generated schedule can be plotted and visualized via the file plotting.py.

#### The RL Trainer and Evaluator

An RL schedule requires an extra training step to generate. The training runner (training.py)
takes the configuration for the observation and the chosen scheduler class (assuming it is RL
compatible) and trains it using the environment supplied in the import statements.

\\ TODO Make the Env and Schedule class a config param

The trainer produces weights saved to the output that can later be used by `model_rollout.
py` to generate a schedule using those weights at a specific time interval, and then plots
visualizations for the output schedule.


<<<<<<< HEAD
<<<<<<< HEAD
A diagram of how these programs interact is given below.
![Todo Provide alt text](https://github.com/deepskies/RLTelescopes/blob/main/figures/Code%20Diagram.png)
=======
=======
>>>>>>> 42f9fd1 (modified start and end date. Added additional pre-commits)
A diagram of how these programs interact is given below.
![Todo Provide alt text](figures/Code Diagram.png)
>>>>>>> 42f9fd1 (modified start and end date. Added additional pre-commits)

## Citation

```
@article{key ,
    author = {You :D},
    title = {title},
    journal = {journal},
    volume = {v},
    year = {20XX},
    number = {X},
    pages = {XX--XX}
}

```

## Acknowledgement
And you <3
