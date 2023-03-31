"""
Load and parse plane data from a .txt file to a numpy array with additional metrics.
"""
import re
import logging
import numpy as np
import random
from PIL import Image, ImageDraw


class PlaneSchedule():
    """
    Load and parse a plance schedule file.
    """
    logger = logging.getLogger(__name__)

    __n_planes = None
    __n_vars = None
    __t_freeze = None
    __raw_data = None
    __norm_data = None
    __draw_data = None

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
        self.__n_planes, self.__t_freeze, self.__raw_data = self.__load_raw__(
            filepath)

        try:
            self.__normalise_data__()
        except ValueError as error:
            self.logger.error("Failed to normalise raw data")
            raise error

    def raw(self):
        """
        Get the raw data loaded from file as a numpy array.
        """
        return self.__raw_data

    def n_planes(self):
        """
        Get the number of planes in this schedule.
        """
        return self.__n_planes

    def n_vars(self):
        return self.__n_vars

    def t_appear(self):
        return self.data()[:,self.COLS["T_APPEAR"]]
    
    def t_early(self):
        return self.data()[:,self.COLS["T_EARLY"]]
    
    def t_target(self):
        return self.data()[:,self.COLS["T_TARGET"]]
    
    def T_assigned(self):
        return self.data()[:,self.COLS["T_ASSIGNED"]]
    
    def t_late(self):
        return self.data()[:,self.COLS["T_LATE"]]
    
    def p_early(self):
        return self.data()[:,self.COLS["P_EARLY"]]
    
    def p_late(self):
        return self.data()[:,self.COLS["P_LATE"]]

    def data(self):
        """
        Get the raw data normalised between 0 and 1 as a numpy array.
        """
        return self.__norm_data
    
    def draw_assigned_times(self, times):
        temp = self.data().copy()
        temp[:, self.COLS["T_ASSIGNED"]] = times
        self.draw_planes(data=temp)

    def draw_data(self):
        """
        Get the draw data of the normalised data.
        """
        return self.__draw_data

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
            self.__n_vars = len(self.COLS) + n_planes
            data = np.zeros([n_planes, self.__n_vars])

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
                    if term_idx == self.__n_vars:
                        plane_idx += 1
                        term_idx = 0

        return n_planes, t_freeze, data

    def __normalise_data__(self):
        if self.__raw_data is None:
            raise ValueError("No raw data is loaded to normalise")

        self.__norm_data = self.__raw_data.copy()

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
                    for i in range(self.__n_planes)])

        for i in range(self.__n_planes):
            self.__norm_data[:, len(self.COLS) + i] = np.interp(
                self.__norm_data[:, len(self.COLS) + i], (i_min, i_max), (0, 1))

    def __generate_draw_data__(self, data, width):
        # if self.__norm_data is None:
        #     raise ValueError("No normalised data is loaded to normalise")

        return (data.copy() * (width - 1)).astype(int)

    def __draw_vert__(self, image, x, row, row_height, gap_height, dotted=False):
        for j in range(row_height-2*gap_height):
            if dotted:
                col = (0, 0, 0) if j % 2 == 0 else (255, 255, 255)
            else:
                col = (0, 0, 0)

            image.putpixel((x, (row*row_height)+gap_height+j), col)

    def __draw_hori__(self, image, x, y, length):
        for j in range(length):
            image.putpixel((x+j, y), (0, 0, 0))

    def mutate(self, prob=1.0, data=None):
        """
        Randomly mutate the schedule
        """
        plane_data = self.__norm_data if data is None else data

        for plane in plane_data:
            if random.random() < prob:
                rand = np.random.uniform(
                    plane[self.COLS["T_EARLY"]], plane[self.COLS["T_LATE"]])
                plane[self.COLS["T_ASSIGNED"]] = rand

        return plane_data

    def evaluate(self, data=None):
        """
        Evaluate how good a given schedule is.
        """
        plane_data = self.__norm_data if data is None else data

        t_delta = plane_data[:, self.COLS["T_ASSIGNED"]] - \
            plane_data[:, self.COLS["T_TARGET"]]
        early_score = np.sum(
            np.where(t_delta < 0, t_delta*plane_data[:, self.COLS["P_EARLY"]], 0))
        late_score = np.sum(
            np.where(t_delta > 0, t_delta*plane_data[:, self.COLS["P_LATE"]], 0))

        # Check all planes are within the landing window.
        constraint_not_early = np.where(plane_data[:, self.COLS["T_ASSIGNED"]] >= plane_data[:, self.COLS["T_EARLY"]], 1, 0)
        constraint_not_late = np.where(plane_data[:, self.COLS["T_ASSIGNED"]] <= plane_data[:, self.COLS["T_LATE"]], 1, 0)
        constraint_within_landing_window = constraint_not_late * constraint_not_early
        constraint_all_planes_within_landing_window = np.any(constraint_within_landing_window)

        return [[early_score, late_score], [constraint_all_planes_within_landing_window]]

    def draw_planes(self, width=512, data=None, pixel_height=20, gap_height=3):
        """
        Draw plane event times for easier analysis of the data.
        """

        if data is None:
            plane_data = self.__generate_draw_data__(self.__norm_data, width)
        else:
            plane_data = self.__generate_draw_data__(data, width)

        row_height = pixel_height+gap_height
        bar_height = pixel_height-gap_height

        image = Image.new('RGB', (width, self.__norm_data.shape[0]*row_height))
        ImageDraw.floodfill(image, xy=(0, 0), value=(255, 255, 255))

        for idx, plane in enumerate(plane_data):
            # Draw appearance time
            self.__draw_vert__(
                image, plane[self.COLS["T_APPEAR"]], idx, row_height, gap_height)

            # Draw appearnce time whisker
            whisker_length = plane[self.COLS["T_EARLY"]
                                   ] - plane[self.COLS["T_APPEAR"]]
            self.__draw_hori__(image, plane[self.COLS["T_APPEAR"]], (
                idx*row_height)+int((pixel_height+gap_height)/2), whisker_length)

            # Draw left (early) and right (late) lines
            self.__draw_vert__(
                image, plane[self.COLS["T_EARLY"]], idx, row_height, gap_height)
            self.__draw_vert__(
                image, plane[self.COLS["T_LATE"]], idx, row_height, gap_height)

            # Draw bars
            bar_length = plane[self.COLS["T_LATE"]] - \
                plane[self.COLS["T_EARLY"]]
            bar_top = (idx*row_height)+gap_height
            self.__draw_hori__(
                image, plane[self.COLS["T_EARLY"]], bar_top, bar_length)
            self.__draw_hori__(
                image, plane[self.COLS["T_EARLY"]], bar_top+bar_height, bar_length)

            # Draw assigned time
            self.__draw_vert__(
                image, plane[self.COLS["T_ASSIGNED"]], idx, row_height, gap_height)

            # Draw target time
            self.__draw_vert__(
                image, plane[self.COLS["T_TARGET"]], idx, row_height, gap_height, dotted=True)

        image.show()
