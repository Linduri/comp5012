"""
Load and parse plane data from a .txt file to a numpy array with additional metrics.
"""
import re
import numpy as np
from typing import List
from pydantic import BaseModel
import logging

class PlaneSchedule():
    logger = logging.getLogger(__name__)

    __n_planes = None
    __t_freeze = None
    __raw_data = None
    __norm_data = None

    COLS = {
        "T_APPEAR": 0,
        "T_EARLY": 1,
        "T_TARGET": 2,
        "T_ASSIGNED": 3,
        "T_LATE": 4,
        "P_EARLY": 5,
        "P_LATE": 6
    }

    def __init__(self, filepath):
        self.__n_planes, self.__t_freeze, self.__raw_data = self.__load_raw__(filepath)

        try:            
            self.__normalise_data__()
        except ValueError as error:
            self.logger.error("Failed to normalise raw data")
            raise error


    def raw(self):
        return self.__raw_data

    def data(self):
        return self.__norm_data

    def __split_terms__(self, line):
        """
        Split a line of numbers into a list of floats
        """
        if line is None:
            return None
        else:
            nums = re.split(r"\s+", line.strip())
            return [float(num) for num in nums]

    def __load_raw__(self, path):
        """
        Load and parse plane data from a .txt file to a numpy array with additional metrics and 
        randomised assigned landing time.
        """
        with open(path, 'r', encoding="utf-8") as file:

            # Get the plane count and freeze time from the first line.
            next_line = file.readline()

            terms = self.__split_terms__(next_line)

            n_planes = int(terms[0])
            t_freeze = terms[1]

            # Keep reading all remaining terms in groups of 6 + n_planes
            # columns = plane attributes + n_planes separation
            # rows = n_planes
            var_per_plane = len(self.COLS) + n_planes
            data = np.zeros([n_planes, var_per_plane])

            term_idx = 0
            plane_idx = 0
            while True:
                next_line = file.readline()

                # Exit if end of file
                if not next_line:
                    break

                for term in self.__split_terms__(next_line):
                    if term_idx == 0:
                        data[plane_idx, self.COLS["T_APPEAR"]] = term
                    elif term_idx == 1:
                        data[plane_idx, self.COLS["T_EARLY"]] = term
                    elif term_idx == 2:
                        data[plane_idx, self.COLS["T_TARGET"]] = term
                    elif term_idx == 3:
                        data[plane_idx, self.COLS["T_LATE"]] = term
                    elif term_idx == 4:
                        data[plane_idx, self.COLS["P_EARLY"]] = term
                    elif term_idx == 5:
                        data[plane_idx, self.COLS["P_LATE"]] = term
                    else:
                        if term_idx == 6:
                            term_idx += 1

                        data[plane_idx, term_idx] = term

                    term_idx += 1
                    if term_idx == var_per_plane:
                        plane_idx += 1
                        term_idx = 0

        return n_planes, t_freeze, data

    def __normalise_data__(self):
        if self.__raw_data is None:
            raise ValueError("No raw data is loaded to normalise")
        
        self.__norm_data = self.__raw_data

        lower_bounds = self.__norm_data.min(axis=0)
        upper_bounds = self.__norm_data.max(axis=0)

        t_earliest = lower_bounds[self.COLS["T_APPEAR"]]
        t_latest = upper_bounds[self.COLS["T_LATE"]]

        #Normalise times
        for time_col in ["T_APPEAR", "T_EARLY", "T_TARGET", "T_ASSIGNED", "T_LATE"]:
            self.__norm_data[:, self.COLS[time_col]] = np.interp(self.__norm_data[:, self.COLS[time_col]], (t_earliest, t_latest), (0, 1))           
