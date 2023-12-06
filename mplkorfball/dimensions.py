"""mplkorfball pitch dimensions.


Map of the pitch dimensions:

(left, top)                                                                             (right, top)
    |---------------------------------------------------------------------------------|
    |                                        |                                        |  ^
    |                                        |                                        |  |
    |                                        |                                        |  |
    |                                        |                                        |  |
    |          ,.*****,.***.          (center_length,)         ,.****.****.           |  |
    |        *     *        `*           center_width)       *        `*    `*        |  |          ^
    |    arc*     *O    '     *              °              *     '    O*     *       |  |  width   | two_fifty_width
    |        *.    *.        *               |               *.        *     *        |  |          v
    |           `*** `**** '`                |                  `*** '`*** '`         |  |
    |                                        |                                        |  |
    |                                        |                                        |  |
    |                                        |                                        |  |
    |                                        |                                        |  v
    |---------------------------------------------------------------------------------|
(left, bottom)                                                                          (right, bottom)

                  | korf_width

     <------------------------------------------------------------------------------->
                                           length
     <------------>
     post_distance

                <-><---->
     korf_length twofifty_length
"""

from dataclasses import dataclass, InitVar

import numpy as np
from typing import Optional


valid = ['fullsize', 'custom']
size_varies = ['custom']


@dataclass
class BaseDims:
    """ Base dataclass to hold pitch dimensions."""
    pitch_width: float
    pitch_length: float
    korf_width: float
    korf_length: float
    twofifty_width: float
    twofifty_length: float
    korf_offset: float
    arc: Optional[float]
    invert_y: bool
    origin_center: bool

    # dimensions that can be calculated in __post_init__
    left: Optional[float] = None
    right: Optional[float] = None
    bottom: Optional[float] = None
    top: Optional[float] = None
    aspect: Optional[float] = None
    width: Optional[float] = None
    length: Optional[float] = None
    post_distance: Optional[float] = None
    post_left: Optional[float] = None
    post_right: Optional[float] = None
    korf_left: Optional[float] = None
    korf_right: Optional[float] = None
    penalty_left: Optional[float] = None
    penalty_right: Optional[float] = None
    penalty_area_top: Optional[float] = None
    penalty_area_bottom: Optional[float] = None
    penalty_area_left: Optional[float] = None
    penalty_area_right: Optional[float] = None
    freepass_left: Optional[float] = None
    freepass_right: Optional[float] = None
    center_width: Optional[float] = None
    center_length: Optional[float] = None

    # defined in pitch_markings
    x_markings_sorted: Optional[np.array] = None
    y_markings_sorted: Optional[np.array] = None
    pitch_extent: Optional[np.array] = None

    def setup_dims(self):
        """ Run methods for the extra pitch dimensions."""
        self.pitch_markings()

    def pitch_markings(self):
        """ Create sorted pitch dimensions to enable standardization of coordinates
         and pitch_extent which contains [xmin, xmax, ymin, ymax]."""
        self.x_markings_sorted = np.array([self.left, self.penalty_area_left, self.post_left,
                                           self.penalty_left, self.freepass_left,
                                           self.center_length,
                                           self.freepass_right, self.penalty_right,
                                           self.post_right, self.penalty_area_right, self.post_right])

        self.y_markings_sorted = np.array([self.bottom, self.penalty_area_bottom,
                                           self.penalty_area_top, self.top])

        if self.invert_y:
            self.y_markings_sorted = np.sort(self.y_markings_sorted)
            self.pitch_extent = np.array([self.left, self.right, self.top, self.bottom])
        else:
            self.pitch_extent = np.array([self.left, self.right, self.bottom, self.top])

    def penalty_area_dims(self):
        """ Create the penalty area dimensions. This is used to calculate the dimensions inside the
        penalty areas for pitches with varying dimensions (width and length varies)."""
        self.penalty_right = self.right - self.penalty_left
        neg_if_inverted = -1 / 2 if self.invert_y else 1 / 2
        self.penalty_area_bottom = self.center_width - (neg_if_inverted * self.twofifty_width)
        self.penalty_area_top = self.center_width + (neg_if_inverted * self.twofifty_width)


@dataclass
class FixedDims(BaseDims):
    """ Dataclass holding the dimensions for pitches with fixed dimensions:
     'fullsize' and ... ."""

    def __post_init__(self):
        self.setup_dims()


@dataclass
class VariableCenterDims(BaseDims):
    """ Dataclass holding the dimensions for pitches where the origin in the center of the pitch:
    'centered' and ... ."""
    post_distance: InitVar[float] = None

    def __post_init__(self, post_distance):
        self.left = - self.pitch_length / 2
        self.right = - self.left
        self.bottom = - self.pitch_width / 2
        self.top = - self.bottom
        self.width = self.pitch_width
        self.length = self.pitch_length
        self.post_left = self.left + post_distance
        self.post_right = - self.post_left
        self.korf_left = self.post_left + self.korf_offset + self.korf_length / 2
        self.korf_right = -self.korf_left
        self.penalty_left = self.post_left + self.twofifty_length
        self.penalty_right = - self.penalty_left
        self.penalty_area_left = self.post_left - self.twofifty_length
        self.penalty_area_right = - self.penalty_area_left
        self.freepass_left = self.penalty_left + self.twofifty_length
        self.freepass_right = - self.freepass_left
        self.setup_dims()


@dataclass
class CustomDims(BaseDims):
    """ Dataclass holding the dimension for the custom pitch.
    This is a pitch where the dimensions (width/length) vary and the origin ins (left, bottom)."""

    def __post_init__(self):

        self.right = self.pitch_length
        self.top = self.pitch_width
        self.post_left = self.left + self.post_distance  # change this to be posts?
        self.center_width = self.pitch_width / 2
        self.center_length = self.pitch_length / 2
        self.korf_left = self.post_left + self.korf_offset + self.korf_length / 2
        self.korf_right = self.right - self.korf_left

        self.penalty_area_dims()
        self.setup_dims()


def fullsize_dims():
    """ Create 'fullsize' dimensions. """
    return FixedDims(left=0., right=100., bottom=0., top=100., aspect=20./40.,
                     width=100., length=100., pitch_width=20., pitch_length=40.,
                     korf_offset=.04, korf_width=.4, korf_length=.4,
                     twofifty_width=2.5, twofifty_length=2.5,
                     post_left=6.67, post_right=33.33, post_distance=6.67,
                     penalty_left=9.17, penalty_right=30.83,
                     penalty_area_bottom=17.5, penalty_area_top=22.5,
                     center_width=20., center_length=20.,
                     arc=90, invert_y=False, origin_center=False)


def custom_dims(pitch_width, pitch_length, post_distance):
    """ Create 'custom' dimensions. """
    return CustomDims(bottom=0., left=0., aspect=1., width=pitch_width, length=pitch_length,
                      pitch_width=pitch_width, pitch_length=pitch_length, twofifty_width=2.5,
                      twofifty_length=2.5, post_distance=post_distance,
                      korf_offset=0.04, korf_width=.4, korf_length=.4,
                      arc=90, invert_y=False, origin_center=False)


def create_pitch_dims(pitch_type, pitch_width=None, pitch_length=None, post_distance=None):
    """ Create pitch dimensions.

        Parameters
        ----------
        pitch_type : str
            The pitch type used in the plot.
            The supported pitch types are: 'fullsize', and 'custom'.
        pitch_length : float, default None
            The pitch length in meters. Only used for the 'custom' pitch_type.
        pitch_width : float, default None
            The pitch width in meters. Only used for the 'custom' pitch_type.
        post_distance: float, default None
            The distance of the post from the back line.
        Returns
        -------
        dataclass
            A dataclass holding the pitch dimensions.
        """
    if pitch_type == 'fullsize':
        return fullsize_dims()
    if pitch_type == 'custom':
        if post_distance is None:
            post_distance = pitch_length / 6
        return custom_dims(pitch_width, pitch_length, post_distance)
    return custom_dims(pitch_width, pitch_length, post_distance)
