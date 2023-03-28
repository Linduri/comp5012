"""
Load and parse plane data from a .txt file to a numpy array with additional metrics.
"""
import re
import logging
import numpy as np
from PIL import Image, ImageDraw

class PlaneSchedule():
    """
    Load and parse a plance schedule file.
    """
    logger = logging.getLogger(__name__)

    __n_planes = None
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

    def data(self):
        """
        Get the raw data normalised between 0 and 1 as a numpy array.
        """
        return self.__norm_data
    
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
        p_max = max(upper_bounds[self.COLS["P_EARLY"]], upper_bounds[self.COLS["P_LATE"]])

        for penalty_col in ["P_EARLY", "P_LATE"]:
            self.__norm_data[:, self.COLS[penalty_col]] = np.interp(
                self.__norm_data[:, self.COLS[penalty_col]], (p_min, p_max), (0, 1))
        # Interval should remain in the same scale across all interval columns.
        i_min = 0
        i_max = max([upper_bounds[len(self.COLS) + i] for i in range(self.__n_planes)])

        for i in range(self.__n_planes):
            self.__norm_data[:, len(self.COLS) + i] = np.interp(
                self.__norm_data[:, len(self.COLS) + i], (i_min, i_max), (0, 1))
    
    def __generate_draw_data__(self, width):
        if self.__norm_data is None:
            raise ValueError("No normalised data is loaded to normalise")

        self.__draw_data = (self.__norm_data.copy() * (width - 1)).astype(int)

        print(self.__draw_data)

    def __draw_vert__(self, image, x, row, row_height, gap_height, dotted=False):
        for j in range(row_height-2*gap_height+1):
                if dotted:
                    col = (0, 0, 0) if j % 2 == 0 else (255, 255, 255)
                else:
                    col = (0, 0, 0)
                
                image.putpixel((x, (row*row_height)+gap_height+j), col)

    def __draw_hori__(self, image, x, y, length):
        for j in range(length):
                image.putpixel((x+j, y), (0, 0, 0))

    def draw_planes(self, pixel_height=20, gap_height=3):
        """
        Draw plane event times for easier analysis of the data.
        """

        width = 512
        self.__generate_draw_data__(width)

        row_height = pixel_height+gap_height
        bar_height = pixel_height-gap_height

        image = Image.new('RGB', (width, self.__norm_data.shape[0]*row_height))
        print(f"{width},{self.__norm_data.shape[0]*row_height}")
        ImageDraw.floodfill(image, xy=(0, 0), value=(255, 255, 255))

        for idx, plane in enumerate(self.__draw_data):
            # Draw appearance time
            self.__draw_vert__(image, plane[self.COLS["T_APPEAR"]], idx, row_height, gap_height)

            # Draw left (early) and right (late) lines
            self.__draw_vert__(image, plane[self.COLS["T_EARLY"]], idx, row_height, gap_height)
            self.__draw_vert__(image, plane[self.COLS["T_LATE"]], idx, row_height, gap_height)

            # Draw bars
            bar_length = plane[self.COLS["T_LATE"]] - plane[self.COLS["T_EARLY"]]
            bar_top = (idx*row_height)+gap_height
            self.__draw_hori__(image, plane[self.COLS["T_EARLY"]], bar_top, bar_length)
            self.__draw_hori__(image, plane[self.COLS["T_EARLY"]], bar_top+bar_height, bar_length)

            # Draw assigned time
            self.__draw_vert__(image, plane[self.COLS["T_ASSIGNED"]], idx, row_height, gap_height)

            # Draw target time
            self.__draw_vert__(image, plane[self.COLS["T_TARGET"]], idx, row_height, gap_height, dotted=True)

        image.show()