"""
Load and parse plane data from a .txt file to a numpy array with additional metrics.
"""
import re
import logging
import random
import numpy as np

from PIL import Image, ImageDraw


class PlaneSchedule():
    """
    Load and parse a plance schedule file.
    """
    logger = logging.getLogger(__name__)

    _n_planes = None
    _n_vars = None
    _t_freeze = None
    _raw_data = None
    _norm_data = None
    _draw_data = None

    COLS = {
        "T_APPEAR": 0,
        "T_EARLY": 1,
        "T_TARGET": 2,
        "T_ASSIGNED": 3,
        "T_LATE": 4,
        "P_EARLY": 5,
        "P_LATE": 6
    }

    def __init__(self, _filepath):
        self._n_planes, self._t_freeze, self._raw_data = self.__load_raw__(
            _filepath)

        try:
            self.__normalise_data__()
        except ValueError as error:
            self.logger.error("Failed to normalise raw data")
            raise error

    def raw(self):
        """
        Get the raw data loaded from file as a numpy array.
        """
        return self._raw_data

    def n_planes(self):
        """
        Get the number of planes in this schedule.
        """
        return self._n_planes

    def n_vars(self):
        """
        Retrieve the scalar number of decision variables per plane.
        """
        return self._n_vars

    def t_freeze(self):
        """
        Retrieve the scalar time within which to freeze a plane.
        """
        return self._t_freeze

    def t_appear(self):
        """
        Retreive the appearance times of each plane as a 1D array.
        """
        return self.data()[:, self.COLS["T_APPEAR"]]

    def t_early(self):
        """
        Retreive the earliest landing times of each plane as a 1D array.
        """
        return self.data()[:, self.COLS["T_EARLY"]]

    def t_target(self):
        """
        Retreive the target landing times of each plane as a 1D array.
        """
        return self.data()[:, self.COLS["T_TARGET"]]

    def t_assigned(self):
        """
        Retreive the assigned landing times of each plane as a 1D array.
        """
        return self.data()[:, self.COLS["T_ASSIGNED"]]

    def t_late(self):
        """
        Retreive the latest landing times of each plane as a 1D array.
        """
        return self.data()[:, self.COLS["T_LATE"]]
    
    def t_separation(self):
        """Retreive the plane separation matrix."""
        return self.data()[:, -(self.n_vars() - len(self.COLS)):]

    def p_early(self):
        """
        Retreive the early landing penalty for each plane as a 1D array.
        """
        return self.data()[:, self.COLS["P_EARLY"]]

    def p_late(self):
        """
        Retreive the late landing penalty for each plane as a 1D array.
        """
        return self.data()[:, self.COLS["P_LATE"]]

    def data(self):
        """
        Get the raw data normalised between 0 and 1 as a numpy array.
        """
        return self.__norm_data

    def draw_assigned_times(self, _times):
        """
        Draw a plane schedule given new landing times and the early and 
        late times for this schuedle.
        """
        temp = self.data().copy()
        temp[:, self.COLS["T_ASSIGNED"]] = _times
        return self.draw_planes(_data=temp)

    def draw_data(self):
        """
        Get the draw data of the normalised data.
        """
        return self.draw_data

    def __split_terms__(self, _line):
        """
        Split a line of numbers into a list of floats
        """
        if _line is None:
            return None
        else:
            nums = re.split(r"\s+", _line.strip())
            return [float(num) for num in nums]

    def __load_raw__(self, _path):
        """
        Load and parse plane data from a .txt file to a numpy array with 
        additional metrics and randomised assigned landing time.
        """
        with open(_path, 'r', encoding="utf-8") as file:

            # Get the plane count and freeze time from the first line.
            next_line = file.readline()

            terms = self.__split_terms__(next_line)

            n_planes = int(terms[0])
            t_freeze = terms[1]

            # Keep reading all remaining terms in groups of 6 + n_planes
            # columns = plane attributes + n_planes separation
            # rows = n_planes
            self._n_vars = len(self.COLS) + n_planes
            data = np.zeros([n_planes, self._n_vars])

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
                    if term_idx == self._n_vars:
                        plane_idx += 1
                        term_idx = 0

        return n_planes, t_freeze, data

    def __normalise_data__(self):
        """
        Map all plane variables to between 0 and 1.
        """
        if self._raw_data is None:
            raise ValueError("No raw data is loaded to normalise")

        self.__norm_data = self._raw_data.copy()

        # Time features should be normalised to the same scale
        # across all time features not per time feature,
        lower_bounds = self.__norm_data.min(axis=0)
        upper_bounds = self.__norm_data.max(axis=0)

        t_earliest = lower_bounds[self.COLS["T_APPEAR"]]
        t_latest = upper_bounds[self.COLS["T_LATE"]]

        for time_col in ["T_APPEAR", "T_EARLY", "T_TARGET", "T_ASSIGNED", "T_LATE"]:
            self.__norm_data[:, self.COLS[time_col]] = np.interp(
                self.__norm_data[:, self.COLS[time_col]], (t_earliest, t_latest), (0, 1))

        # Penalty scores should remain in the same scale across all penalty columns.
        p_min = 0
        p_max = max(upper_bounds[self.COLS["P_EARLY"]],
                    upper_bounds[self.COLS["P_LATE"]])

        for penalty_col in ["P_EARLY", "P_LATE"]:
            self.__norm_data[:, self.COLS[penalty_col]] = np.interp(
                self.__norm_data[:, self.COLS[penalty_col]], (p_min, p_max), (0, 1))
        # Interval should remain in the same scale across all interval columns.
        i_min = 0
        i_max = max([upper_bounds[len(self.COLS) + i]
                    for i in range(self._n_planes)])

        for i in range(self._n_planes):
            self.__norm_data[:, len(self.COLS) + i] = np.interp(
                self.__norm_data[:, len(self.COLS) + i], (i_min, i_max), (0, 1))

    def __generate_draw_data__(self, _data, _width):
        """
        Generate pixel coordianates for each variable.
        """
        return (_data.copy() * (_width - 1)).astype(int)

    def __draw_vert__(self, image, _x, _row, _row_height, _gap_height, dotted=False):
        """
        Draw a vertical line.
        """
        for j in range(_row_height-2*_gap_height):
            if dotted:
                col = (0, 0, 0) if j % 2 == 0 else (255, 255, 255)
            else:
                col = (0, 0, 0)

            image.putpixel((_x, (_row*_row_height)+_gap_height+j), col)

    def __draw_hori__(self, image, _x, _y, _length):
        """
        Draw a horizontal line.
        """
        for j in range(_length):
            image.putpixel((_x+j, _y), (0, 0, 0))

    def mutate(self, _prob=1.0, _data=None):
        """
        Randomly mutate the schedule
        """
        plane_data = self.__norm_data if _data is None else _data

        for plane in plane_data:
            if random.random() < _prob:
                rand = np.random.uniform(
                    plane[self.COLS["T_EARLY"]], plane[self.COLS["T_LATE"]])
                plane[self.COLS["T_ASSIGNED"]] = rand

        return plane_data

    def draw_planes(self, _width=512, _data=None, _pixel_height=20, _gap_height=3):
        """
        Draw plane event times for easier analysis of the data.
        """

        if _data is None:
            plane_data = self.__generate_draw_data__(self.__norm_data, _width)
        else:
            plane_data = self.__generate_draw_data__(_data, _width)

        row_height = _pixel_height+_gap_height
        bar_height = _pixel_height-_gap_height

        image = Image.new('RGB', (_width, self.__norm_data.shape[0]*row_height))
        ImageDraw.floodfill(image, xy=(0, 0), value=(255, 255, 255))

        for idx, plane in enumerate(plane_data):
            # Draw appearance time
            self.__draw_vert__(
                image, plane[self.COLS["T_APPEAR"]], idx, row_height, _gap_height)

            # Draw appearnce time whisker
            whisker_length = plane[self.COLS["T_EARLY"]
                                   ] - plane[self.COLS["T_APPEAR"]]
            self.__draw_hori__(image, plane[self.COLS["T_APPEAR"]], (
                idx*row_height)+int(row_height/2), whisker_length)

            # Draw left (early) and right (late) lines
            self.__draw_vert__(
                image, plane[self.COLS["T_EARLY"]], idx, row_height, _gap_height)
            self.__draw_vert__(
                image, plane[self.COLS["T_LATE"]], idx, row_height, _gap_height)

            # Draw bars
            bar_length = plane[self.COLS["T_LATE"]] - \
                plane[self.COLS["T_EARLY"]]
            bar_top = (idx*row_height)+_gap_height
            self.__draw_hori__(
                image, plane[self.COLS["T_EARLY"]], bar_top, bar_length)
            self.__draw_hori__(
                image, plane[self.COLS["T_EARLY"]], bar_top+bar_height, bar_length)

            # Draw assigned time
            self.__draw_vert__(
                image, plane[self.COLS["T_ASSIGNED"]], idx, row_height, _gap_height)

            # Draw target time
            self.__draw_vert__(
                image, plane[self.COLS["T_TARGET"]], idx, row_height, _gap_height, dotted=True)

        return image
