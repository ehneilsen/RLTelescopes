from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import pytest

from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import palpy
from rltelescope.low_airmass_scheduler import SqueScheduler


def test_airmass_scheduler():
    # Replicate the behaviour of parsing arguments
    config_dir = Path(__file__).parents[1].joinpath("rltelescope", "train_configs")
    obsprog_config = config_dir.joinpath("default_obsprog.conf")
    schedule_config = config_dir.joinpath("pure_airmass_schedule.config")
    start_date = "2021-09-08T20:00:00Z"
    end_date = "2021-09-11T20:00:00Z"

    with TemporaryDirectory(prefix="test_airmass_scheduler_") as out_path:
        arguments = argparse.Namespace(
            schedule_config=schedule_config,
            obsprog_config=obsprog_config,
            start_date=start_date,
            end_date=end_date,
            out_path=out_path,
        )

        scheduler = SqueScheduler(arguments.schedule_config, arguments.obsprog_config)
        scheduler.update(arguments.start_date, arguments.end_date)
        scheduler.save(arguments.out_path)

    field_eq = SkyCoord(
        ra=scheduler.actions.ra,
        dec=scheduler.actions.decl,
        unit="deg",
        frame="icrs",
        location=scheduler.obsprog.observatory.location,
    )

    for _, obs in scheduler.schedule.iterrows():
        # Hack to get the MJD for which the airmass was calculated.
        # Guess that the time for this exposure is the end time of
        # the previous one, which is what is recorded in obs.mjd
        try:
            start_mjd = previous_obs_end_mjd
        except NameError:
            start_mjd = Time(start_date, scale='utc').mjd

        previous_obs_end_mjd = obs.mjd

        obs_time = Time(start_mjd, format="mjd", scale="utc")

        field_altaz = field_eq.transform_to(
            AltAz(obstime=obs_time, location=scheduler.obsprog.observatory.location)
        )
        palpy_airmass = palpy.airmas((90 * u.deg - field_altaz.alt.max()).rad)
        assert obs.reward == pytest.approx(palpy_airmass, rel=0.01, abs=0.01)
