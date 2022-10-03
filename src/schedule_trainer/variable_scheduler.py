"""
A stealth copy of delve's obztrak
Uses the same variables to select an action
based on the best choices at any given time

"""
import pandas as pd
from src.schedule_trainer.scheduler import Scheduler


class VariableScheduler(Scheduler):
    def __init__(self, config):
        super().__init__(config)

    def update(self, obsprog):
        obsprog.reset()
        done = False
        while not done:
            action = self.calculate_action(obsprog=obsprog)
            self.feed_action(action)
            reward = self.reward(obsprog.state)

            self.update_schedule(action, reward)
            done = self.check_endtime(obsprog, action)

    @staticmethod
    def quality(new_obs):
        slew_cubed = new_obs['slew']**3
        hour_angle = new_obs['ha']
        airmass_cubed = 100*(new_obs['airmass']-1)**3
        quality = slew_cubed + hour_angle + airmass_cubed
        return quality

    def calculate_action(self, **action_params):
        obsprog = action_params["obsprog"]
        # Iterate through actions
        valid_actions = []
        quality = []
        for action in self.actions:
            new_obs = obsprog.calculate_observation(action)
            quality.append(VariableScheduler.quality(new_obs=new_obs))
            valid_actions.append(obsprog.valid_action(new_obs))

        actions = self.actions.copy()
        actions['quality'] = pd.Series(quality)
        valid_actions = self.actions[pd.Series(valid_actions)]
        action = valid_actions.argmax()

        return action
