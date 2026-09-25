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

    def __init__(self, lambda_liv, n_liv, nu_syn):
        self.lambda_liv = lambda_liv
        self.n_liv = n_liv
        self.nu_syn = nu_syn

    def calculate_time_delay (self, energy):
        """Calculates the time delay
        """
        return self.lambda_liv * energy**self.n_liv


    # def functions that will be needed like Doppler boosting? 
    # def plot function?
    # def lightcurve function?
