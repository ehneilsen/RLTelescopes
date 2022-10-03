"""
An adaptation of the observation program written by Eric Neilson in Astrotact:  https://github.com/ehneilsen/astrotact.git

Designed to step through the night and calculate variables for a given point site/calculation combo.
"""
import os.path

import pandas as pd
import configparser
import ast
import numpy as np
import numexpr

from astropy.coordinates import get_sun, get_moon
import astroplan
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates.earth import EarthLocation
from skybright import skybright

class ObservationProgram:
    def __init__(self, start_date, config_path):
        self.config = configparser.ConfigParser()
        assert os.path.exists(config_path)
        self.config.read(config_path)

        self.observatory = self.init_observatory()
        self.start_time = self.init_time_start(start_date)
        self.slew_rate = self.init_slew()
        self.calc_sky = skybright.MoonSkyModel(self.config)
        self.set_optics()

        self.mjd = self.start_time
        self.decl = 0
        self.ra = 0
        self.band = "g"
        self.exposure_time = 300

        self.obs = self.observation()
        self.state = self.exposures()

    def set_optics(self):
        self.optics_fwhm = self.config.getfloat('optics', 'fwhm')
        try:
            self.band_wavelength = ast.literal_eval(
                    self.config.get('bands', 'wavelengths'))
        except KeyError:
            self.band_wavelength = {
                'u': 380.0,
                'g': 475.0,
                'r': 635.0,
                'i': 775.0,
                'z': 925.0,
                'Y': 1000.0
            }

        self.wait_time = 300
        try:
            self.filter_change_time = self.config.getfloat("bands",
                                                   "filter_change_rate")
        except KeyError:
            self.filter_change_time = 0.0


    def observation(self):
        return {
            "mjd": self.mjd,
            "decl":self.decl,
            "ra": self.ra,
            "band": self.band,
            "exposure_time": self.exposure_time
        }

    def reset(self):
        self.mjd = self.start_time
        self.decl = 0
        self.ra = 0
        self.band = 'g'

        self.obs = self.observation()
        self.state = self.exposures()

    def init_observatory(self):
        lat = self.config.getfloat('Observatory Position', 'latitude')
        long = self.config.getfloat('Observatory Position', 'longitude')
        ele = self.config.getfloat('Observatory Position', 'elevation')

        return astroplan.Observer(
            longitude=long*u.deg,
            latitude=lat*u.deg,
            elevation=ele*u.m
        )

    def init_time_start(self, start_date):
        return Time(start_date, location=self.observatory.location).mjd

    def init_slew(self):
        slew_expr = self.config.getfloat('slew', 'slew_expr')
        return slew_expr

    @staticmethod
    def calc_airmass(hzcrds=None, zd=None):
        if hzcrds is not None:
            cos_zd = np.cos(np.radians(90) - hzcrds.alt.rad)
        else:
            cos_zd = np.cos(np.radians(zd))

        a = numexpr.evaluate("462.46 + 2.8121/(cos_zd**2 + 0.22*cos_zd + 0.01)")
        airmass = numexpr.evaluate("sqrt((a*cos_zd)**2 + 2*a + 1) - a * cos_zd")
        airmass[hzcrds.alt.rad < 0] = np.nan
        return airmass

    def current_coord(self):
        return SkyCoord(ra=self.ra*u.degree, dec=self.decl*u.degree)

    def trans_to_altaz(self, coord):
        alt_az = self.observatory.altaz(time=Time(self.mjd, format='mjd'),
                                        target=coord)
        return alt_az

    def exposures(self, seeing=0.9, clouds=0.0):
        ANGLE_UNIT = u.deg
        RIGHT_ANGLE = (90 * u.deg).to_value(ANGLE_UNIT)

        time = Time(self.mjd, format='mjd')

        exposure = {}
        exposure['seeing'] = seeing
        exposure['clouds'] = clouds

        exposure['lst'] = self.observatory.local_sidereal_time(time,
                                                               'mean').to_value(
            ANGLE_UNIT)

        hzcrds = self.trans_to_altaz(self.current_coord())
        exposure['az'] = hzcrds.az.to_value(ANGLE_UNIT)
        exposure['alt'] = hzcrds.alt.to_value(ANGLE_UNIT)
        exposure['zd'] = RIGHT_ANGLE - exposure['alt']
        exposure['ha'] = exposure['lst'] - self.ra
        exposure['airmass'] = ObservationProgram.calc_airmass(hzcrds)

        # Sun coordinates
        sun_crds = get_sun(time)
        exposure['sun_ra'] = sun_crds.ra.to_value(ANGLE_UNIT)
        exposure['sun_decl'] = sun_crds.dec.to_value(ANGLE_UNIT)
        sun_hzcrds = self.trans_to_altaz(sun_crds)
        exposure['sun_az'] = sun_hzcrds.az.to_value(ANGLE_UNIT)
        exposure['sun_alt'] = sun_hzcrds.alt.to_value(ANGLE_UNIT)
        exposure['sun_zd'] = RIGHT_ANGLE - exposure['sun_alt']
        exposure['sun_ha'] = exposure['lst'] - exposure['sun_ra']

        # Moon coordinates
        site_location = self.observatory.location
        moon_crds = get_moon(location=site_location, time=time)
        exposure['moon_ra'] = moon_crds.ra.to_value(ANGLE_UNIT)
        exposure['moon_decl'] = moon_crds.dec.to_value(ANGLE_UNIT)
        moon_hzcrds = self.observatory.moon_altaz(time)
        exposure['moon_az'] = moon_hzcrds.az.to_value(ANGLE_UNIT)
        exposure['moon_alt'] = moon_hzcrds.alt.to_value(ANGLE_UNIT)
        exposure['moon_zd'] = RIGHT_ANGLE - exposure['moon_alt']
        exposure['moon_ha'] = exposure['lst'] - exposure['moon_ra']
        exposure['moon_airmass'] = ObservationProgram.calc_airmass(moon_hzcrds)

        # Moon phase
        exposure['moon_phase'] = astroplan.moon.moon_phase_angle(time)
        exposure['moon_illu'] = self.observatory.moon_illumination(time)
        # Moon brightness
        moon_elongation = moon_crds.separation(sun_crds)
        alpha = 180.0 - moon_elongation.deg
        # Allen's _Astrophysical Quantities_, 3rd ed., p. 144
        exposure['moon_Vmag'] = -12.73 \
                                + 0.026 * np.abs(alpha) + 4E-9 * (alpha ** 4)

        exposure['moon_angle'] = (moon_crds
                                  .separation(self.current_coord())
                                  .to_value(ANGLE_UNIT))

        exposure['sky_mag'] = self.calc_sky(
            self.mjd,
            self.ra,
            self.decl,
            self.band,
            moon_crds=moon_crds,
            moon_elongation=moon_elongation.deg,
            sun_crds=sun_crds)

        m0 = self.calc_sky.m_zen[self.band]

        nu = 10 ** (-1 * clouds / 2.5)

        pt_seeing = seeing * exposure['airmass'] ** 0.6
        fwhm500 = np.sqrt(pt_seeing ** 2 + self.optics_fwhm ** 2)

        wavelength = self.band_wavelength[self.band]
        band_seeing = pt_seeing * (500.0 / wavelength) ** 0.2
        fwhm = np.sqrt(band_seeing ** 2 + self.optics_fwhm ** 2)
        exposure['fwhm'] = fwhm

        exposure['tau'] = ((nu * (0.9 / fwhm500)) ** 2) \
                          * (10 ** ((exposure['sky_mag'] - m0) / 2.5))

        exposure['tau'] = 0.0 if ~np.isfinite(exposure['tau']) else \
            exposure['tau']

        exposure['teff'] = exposure['tau'] * self.exposure_time

        for key in self.obs:
            exposure[key] = self.obs[key]

        self.state = exposure
        return exposure

    def update_mjd(self, mjd, ra, decl, band):

        if mjd is None:
            original_coords = self.current_coord()

            if (ra is None) and (decl is None):
                updated_coords = original_coords

            elif (ra is not None) and (decl is None):
                updated_coords = SkyCoord(ra=ra * u.degree, dec=self.decl *
                                                                u.degree)
            elif (ra is None) and (decl is not None):
                updated_coords = SkyCoord(ra=self.ra * u.degree, dec=decl *
                                                                 u.degree)
            else:
                updated_coords = SkyCoord(ra=ra * u.degree, dec=decl * u.degree)

            coord_sep = original_coords.separation(updated_coords)
            slew_time = self.slew_rate * coord_sep/ u.degree

            if slew_time == 0.0:
                slew_time = self.wait_time
            if band != self.band:
                slew_time += self.filter_change_time

            self.mjd += slew_time/(60*24) # Convert to days

        else:
            self.mjd = mjd

    def update_observation(self, mjd=None, ra=None, decl=None, band=None,
                           exposure_time=None):
        # Updates the observation based on input. Any parameters not given
        # are held constant

        # Todo update with timestep and not just the action
        # Dependent on the earth's rotation.
        # Slightly more realistic, not strictly necessary for this use case
        self.update_mjd(mjd, ra, decl, band)

        self.ra = ra if ra is not None else self.ra
        self.decl = decl if decl is not None else self.decl
        self.band = band if band is not None else self.band
        self.exposure_time = exposure_time if exposure_time is not None else \
            self.exposure_time

        self.obs = self.observation()
        self.state = self.exposures()


