"""
Takes model weights that produce a schedul
and produces a schedule at a specific time for it.

Makes the assumption that the schedule is producing weights for an equation;
Producing an equation for each step.
"""
import os
import ray.rllib.agents.es as es
from astropy.time import Time

class ModelRollout:
    def __init__(self, experiment_path, scheduler, environment, env_config):
        self.experiment_path = experiment_path
        self.model_path = self.get_model_path()

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

    def get_model_path(self):
        checkpoint_path = f"{self.experiment_path}/checkpoints"
        checkpoints = os.listdir(checkpoint_path)
        last_checkpoint = checkpoints[-1]
        name = f"checkpoint-{last_checkpoint.split('0')[-1]}"

        checkpoint_path = f"{checkpoint_path}/{last_checkpoint}/{name}"
        assert os.path.exists(checkpoint_path)

        return checkpoint_path

    def load_model(self):
        model_path = self.get_model_path()

        def load_agent():
            agent_config = es.DEFAULT_CONFIG.copy()
            agent_config["env_config"] = self.env_config
            agent_config['num_workers'] = 1
            agent_config['episodes_per_batch'] = 10
            agent_config["evaluation_duration"] = 10
            agent_config['recreate_failed_workers'] = False
            agent = es.ESTrainer(config=agent_config, env=self.environment)

            return agent

        agent = load_agent()
        agent.restore(model_path)
        return agent

    def step_model(self, agent, previous_state):
        action_weights = agent.compute_single_action(previous_state)
        self.scheduler.update(action_weights)
        state = self.scheduler.state
        done = self.scheduler.check_endtime()
        return state, done

    def generate_schedule(self, start_date, end_date, save=False):
        start_time = Time(start_date, format='isot').mjd
        end_time = Time(end_date, format='isot').mjd

        self.step_env.reset()

        self.step_env.mjd = start_time
        self.step_env.start_time, self.step_env.end_time = start_time, end_time

        self.step_env.obs = self.step_env.scheduler.obsprog.observation()
        self.step_env.state = self.step_env.scheduler.obsprog.exposures()

        state = self.step_env.state
        agent = self.load_model()
        done = False
        while not done:
            state, done = self.step_model(agent, state)

        if save:
            self.scheduler.save(self.experiment_path)

        return self.get_schedule()



