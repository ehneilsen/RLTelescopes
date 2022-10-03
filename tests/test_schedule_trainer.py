from unittest import TestCase
import unittest
import sys
import os

sys.path.append("..")
from src.schedule_trainer.scheduler import Scheduler
from src.schedule_trainer.variable_scheduler import VariableScheduler
from src.schedule_trainer.rl_scheduler import RLScheduler
from src.schedule_trainer.squentical_scheduler import SqueScheduler
from src.schedule_trainer.observation_program import ObservationProgram

from astropy.time import Time
import astroplan

scheduler_config_path = os.path.abspath("../src/schedule_trainer/train_configs"
                                        "/default_schedule.conf")
obs_config_path = os.path.abspath("../src/schedule_trainer/train_configs" \
                              "/default_obsprog.conf")
agent_config_path = os.path.abspath("../src/schedule_trainer/train_configs" \
                              "/default_agent.conf")


class TestObsprog(TestCase):
    def setUp(self) -> None:
        start_time = "2018-09-16T01:00:00Z"
        self.obsprog = ObservationProgram(start_time, obs_config_path)

    def test_set_observatory(self):
        self.assertEquals(type(self.obsprog.observatory), astroplan.Observer)

    def test_obprog_reset(self):
        expected_default = {
                "mjd": Time("2018-09-16T01:00:00Z",
                            location=self.obsprog.observatory.location).mjd,
                "decl": 0,
                "ra": 0,
                "band": "g",
                "exposure_time": 300
        }
        for _ in range(5):
            self.obsprog.update_observation()

        self.assertNotEquals(expected_default["mjd"],
                             self.obsprog.obs["mjd"])

        self.obsprog.reset()
        observation = self.obsprog.obs
        for key in expected_default:
            self.assertEquals(expected_default[key], observation[key])

    def test_obprog_step_obs(self):
        init_time = self.obsprog.obs["mjd"]
        self.obsprog.update_observation()

        update_time = self.obsprog.obs["mjd"]
        self.assertTrue(update_time > init_time)
        self.obsprog.reset()

    def test_obs_step_state(self):
        ## So I don't actually care to walk through each variable and
        # calculate it by hand
        # So I'll just verify they're not the same (thus it's updating)
        init_state = self.obsprog.state
        self.obsprog.update_observation()
        next_state = self.obsprog.state

        self.assertNotEqual(init_state, next_state)


class TestScheduler(TestCase):
    def setUp(self) -> None:
        self.scheduler = Scheduler(scheduler_config_path)

    def test_make_action_table(self):
        actions = self.scheduler.generate_action_table()

        expected_columns = {"ra", "decl", "band", "exposure_time"}
        columns = set(actions.columns)
        self.assertEqual(expected_columns, columns)

        self.assertEqual(len(actions), 10)

    def test_teff_reward(self):
        observation = ObservationProgram("2018-09-16T01:00:00Z",
                                         obs_config_path).state
        invalid_reward = self.scheduler.invalid_reward
        self.assertEqual(invalid_reward, self.scheduler.reward(observation))

    def test_invalid_actions(self):
        observation = ObservationProgram("2018-09-16T01:00:00Z",
                                         obs_config_path).state
        self.assertTrue(observation)

    def test_save_schedule(self):
        self.scheduler.save(".")

        self.assertTrue(os.path.exists("./schedule.csv"))
        if os.path.exists("./schedule.csv"):
            os.remove("./schedule.csv")


class TestRLScheduler(TestCase):
    def setUp(self) -> None:
        pass

    def test_default_prediction(self):
        pass

    def test_update(self):
        pass

    def test_init_random_time(self):
        pass


class TestSquenScheduler(TestCase):
    def setUp(self) -> None:
        pass

class TestVarScheduler(TestCase):
    def setUp(self) -> None:
        self.scheduler = VariableScheduler(scheduler_config_path)

    def test_calculate_actions(self):
        pass

    def test_update_schedule(self):
        pass



if __name__ == '__main__':
    unittest.main()