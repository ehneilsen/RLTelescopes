from unittest import TestCase
import unittest
import sys
import os

sys.path.append("..")
from src.schedule_trainer.scheduler import Scheduler
from src.schedule_trainer.rl_agent import RLAgent
from src.schedule_trainer.observation_program import ObservationProgram

import astropy.units as u
from astropy.time import Time
import astroplan


class TestObsprog(TestCase):
    def setUp(self) -> None:
        config_path = os.path.abspath("../src/schedule_trainer/train_configs" \
                              "/default_obsprog.conf")
        start_time = "2018-09-16T01:00:00Z"
        self.obsprog = ObservationProgram(start_time, config_path)

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

class TestRLAgent(TestCase):
    def setUp(self) -> None:

        pass

    def test_update_eq(self):
        pass

    def test_make_action_table(self):
        pass

    def test_teff_reward(self):
        pass

    def test_pick_actions(self):
        pass

    def test_invalid_actions(self):
        pass

    def test_save_schedule(self):
        pass



class TestScheduler(TestCase):
    def setUp(self) -> None:
        pass

    def test_default_prediction(self):
        pass

    def test_update(self):
        pass

    def test_init_random_time(self):
        pass


if __name__ == '__main__':
    unittest.main()