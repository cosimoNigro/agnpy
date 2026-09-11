import logging
import numpy as np
import astropy.units as u
from astropy.constants import h
from agnpy.time_evolution import TimeEvolution, synchrotron_loss


# Idea = take the lambda parameter as input, calculate the delay induced on each flux point (?) and either calculate the spectrum as a function of time or introduce a parameter of the class representing the time after the start of the inpulse


class LIVdelay:
    """
    Injects LIV-type delays into spectrum calculation

    Parameters
    ----------


    """

    def __init__(self, lambda_liv, n_liv, blob, nu_syn, synch, time_step, steps):
        self.lambda_liv = lambda_liv
        self.n_liv = n_liv
        self.blob = blob
        self.synch = synch
        self.time_step = time_step
        self.steps = steps
        self.nu_syn = nu_syn
        self.time_evoluted_seds = self.calculate_time_evoluted_seds() # so a 2D matrix, containing the spectral evolution as a function of time

    def calculate_time_evoluted_seds(self):
        """
        Calculates the array of time evoluted SEDs
        """
        time_evoluted_seds = []
        for i in range(self.steps):
            time_evolution = TimeEvolution(self.blob, self.time_step, synchrotron_loss(self.synch))
            time_evolution_result = time_evolution.evaluate()
            time_evoluted_seds.append(self.synch.sed_flux(self.nu_syn))
        return np.array(time_evoluted_seds)

    def interpolate_seds(self, nu_index, t):
        """
        Interpolates SEDs in between time frames
        """
        time_obs_array = np.linspace(0, self.steps*self.time_step, self.steps+1)
        time_index = np.searchsorted(time_obs_array, t, side='left')
        slope = (self.time_evoluted_seds[time_index+1][nu_index]-self.time_evoluted_seds[time_index][nu_index]) / self.time_step
        return self.time_evoluted_seds[time_index][nu_index] + slope * (t-time_obs_array[time_index])

    def evaluate(self, time_obs):
        """
        Evaluates the SED for a given time of observation
        """
        
        time_evoluted_sed_liv = np.zeros(len(self.nu_syn))
        for nu_index, nu in enumerate(self.nu_syn):
            time_obs_liv = time_obs + self.lambda_liv * (nu*h.to("TeV s"))**self.n_liv # change it to avg frequency in the bin
            time_evoluted_sed_liv[nu_index] = self.interpolate_seds(nu_index, time_obs_liv)

        return time_evoluted_sed_liv


    # def functions that will be needed like Doppler boosting? 
