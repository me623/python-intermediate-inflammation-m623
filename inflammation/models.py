"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row
contains inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :return: loaded text
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: 2D array for mean calculation
    :return: mean value of day column
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2D array for max calulation
    :return: max value of day column
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: 2D array for max calulation
    :return: min value of day column
    :return: min value of day
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    '''Normalise patient data from 2D array of inflammation data'''
    if np.any(data < 0):
        raise ValueError("Inflammation values cant be negative")

    maxes = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / maxes[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised


class Observation:
    def __init__(self, day, value):
        self.day = day
        self.value = value

    def __str__(self):
        return str(self.value)


class Person:
    """Person with name"""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


# Patient class with add_observation method
class Patient(Person):
    """Person with observation
    """

    def __init__(self, name: str, observations=[], placebo=None):
        super().__init__(name)
        self.observations = observations
        self.placebo = placebo
    @property
    def last_observation(self):
        return self.observations[-1]

    def add_observation(self, value, day=None):
        if day is None:
            try:
                day = self.observations[-1].day + 1
            except IndexError:
                day = 0

        new_observation = Observation(day, value)

        self.observations.append(new_observation)
        return new_observation


class Doctor(Person):
    """ A Person with Patients"""

    def __init__(self, name: str, patients: Patient = []):
        super().__init__(name)
        self.patients = patients

    def __str__(self):
        return ("{} {}".format(self.name, self.patients))

    def add_patient(self, patient: Patient):
        new_patient = Patient(patient)
        self.patients.append(new_patient)
        return new_patient

def standard_deviation(data):
    # calculates std of column
    return np.std(data, axis=0)