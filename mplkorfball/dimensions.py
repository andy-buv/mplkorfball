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


valid = ['ikf', 'custom']
size_varies = ['custom']


@dataclass
class KorfballBaseDims:
    """ Base dataclass to hold pitch dimensions."""
    pitch_width: float
    pitch_length: float
    korf_offset: float
    korf_width: float
    korf_length: float
    twofifty_width: float
    twofifty_length: float
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
                                           self.post_right, self.penalty_area_right, self.right])

        self.y_markings_sorted = np.array([self.bottom, self.penalty_area_bottom, self.center_width,
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
        neg_if_inverted = -1 if self.invert_y else 1
        self.penalty_area_bottom = self.center_width - (neg_if_inverted * self.twofifty_width)
        self.penalty_area_top = self.center_width + (neg_if_inverted * self.twofifty_width)


@dataclass
class FixedDimsKorfball(KorfballBaseDims):
    """ Dataclass holding the dimensions for pitches with fixed dimensions:
     'ikf' and ... ."""

    def __post_init__(self):
        self.setup_dims()


@dataclass
class VariableCenterDimsKorfball(KorfballBaseDims):
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
class CustomDimsKorfball(KorfballBaseDims):
    """ Dataclass holding the dimension for the custom pitch.
    This is a pitch where the dimensions (width/length) vary and the origin ins (left, bottom)."""
    post_distance: InitVar[float] = None

    def __post_init__(self, post_distance):

        self.right = self.pitch_length
        self.top = self.pitch_width
        self.post_left = self.left + post_distance  # TODO change this to be posts?
        self.post_right = self.right - self.post_left
        self.center_width = self.pitch_width / 2
        self.center_length = self.pitch_length / 2
        self.korf_left = self.post_left + self.korf_offset + self.korf_length / 2
        self.korf_right = self.right - self.korf_left
        self.penalty_left = self.post_left + self.twofifty_length
        self.penalty_area_dims()
        self.setup_dims()


def ikf_dims():
    """ Create 'ikf' dimensions. """
    return FixedDimsKorfball(pitch_width=20., pitch_length=40.,
                             korf_offset=.04, korf_width=.4, korf_length=.4,
                             twofifty_width=2.5, twofifty_length=2.5, arc=90,
                             invert_y=False, origin_center=False,
                             left=0., right=40., bottom=0., top=20., aspect=1.,
                             width=20., length=40., post_distance=6.67,
                             post_left=6.67, post_right=33.33, korf_left=6.9, korf_right=33.50,
                             penalty_left=9.17, penalty_right=30.83,
                             penalty_area_top=12.5, penalty_area_bottom=7.5,
                             penalty_area_left=4.17, penalty_area_right=15.83,
                             freepass_left=11.67, freepass_right=28.33,
                             center_width=10., center_length=20.,
                             )


def custom_dims(pitch_width, pitch_length, post_distance):
    """ Create 'custom' dimensions. """
    return CustomDimsKorfball(bottom=0., left=0., aspect=1., width=pitch_width, length=pitch_length,
                              pitch_width=pitch_width, pitch_length=pitch_length, twofifty_width=2.5,
                              twofifty_length=2.5, post_distance=post_distance,
                              korf_offset=0.12, korf_width=.4, korf_length=.4,
                              arc=90, invert_y=False, origin_center=False)


def create_pitch_dims(pitch_type, pitch_width=None, pitch_length=None, post_distance=None):
    """ Create pitch dimensions.

        Parameters
        ----------
        pitch_type : str
            The pitch type used in the plot.
            The supported pitch types are: 'ikf', and 'custom'.
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
    if pitch_type == 'netball':
        return netball_dims()
    if pitch_type == 'ikf':
        return ikf_dims()
    if pitch_type == 'custom':
        if post_distance is None:
            post_distance = pitch_length / 6
        return custom_dims(pitch_width, pitch_length, post_distance)
    return custom_dims(pitch_width, pitch_length, post_distance)


"""

Map of the netball pitch dimensions:

(left, top)                                                                             (right, top)
    |---------------------------|---------------------------|---------------------------|  ^
    |                           |                           |                           |  |
    |                           |                           |                           |  |
    |*** .                      |                           |                     .  ***|  |
    |       *                   | (center_length,           |                   *       |  |
    |         *                 |      center_width)        |                 *         |  | width
    |                           |            .-.            |                           |  |
    |o         *                |           *   *           |                *         o|  |       
    |                           |            `-`            |                           |  | ^
    |         *                 |                           |                 *         |  | | circle_width
    |       *                   |                           |                   *       |  | | 
    |***  '                     |                           |                      ` ***|  | v
    |                           |                           |                           |  |
    |                           |                           |                           |  |
    |---------------------------|---------------------------|---------------------------|  v
(left, bottom)                                                                          (right, bottom)
    <--------->                              <->
    circle_length                       center_diameter
                                 <------------------------->             
                                    third_length
     <------------------------------------------------------------------------------->
                                           length
"""

@dataclass
class NetballBaseDims:
    pitch_width: float
    pitch_length: float
    circle_length: float
    circle_width: float
    third_length: float
    center_diameter: float
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

    center_width: Optional[float] = None
    center_length: Optional[float] = None
    third_left: Optional[float] = None
    third_right: Optional[float] = None

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
        self.x_markings_sorted = np.array([self.left, self.third_left,
                                           self.center_length,
                                           self.third_right, self.right])

        self.y_markings_sorted = np.array([self.bottom, self.center_width, self.top])

        if self.invert_y:
            self.y_markings_sorted = np.sort(self.y_markings_sorted)
            self.pitch_extent = np.array([self.left, self.right, self.top, self.bottom])
        else:
            self.pitch_extent = np.array([self.left, self.right, self.bottom, self.top])

@dataclass
class FixedDimsNetball(NetballBaseDims):
    """ Dataclass holding the dimensions for pitches with fixed dimensions:
         'ikf' and ... ."""
    def __post_init__(self):

        self.center_width = self.pitch_width / 2
        self.center_length = self.pitch_length / 2
        self.third_left = self.left + self.third_length
        self.third_right = self.right - self.third_length
        self.setup_dims()


@dataclass
class VariableCenterDimsNetball(NetballBaseDims):
    """ Dataclass holding the dimensions for pitches where the origin in the center of the pitch:
        'centered' """

    def __post_init__(self):
        self.left = - self.pitch_length / 2
        self.right = - self.left
        self.bottom = - self.pitch_width / 2
        self.top = - self.bottom
        self.width = self.pitch_width
        self.length = self.pitch_length
        self.third_left = - self.pitch_length / 6
        self.third_right = - self.third_right
        self.setup_dims()


def netball_dims():

    return FixedDimsNetball(pitch_length=30.5, pitch_width=15.25,
                            circle_length=4.9, circle_width=4.9,
                            third_length=10.167, center_diameter=.9,
                            invert_y=False, origin_center=False)

