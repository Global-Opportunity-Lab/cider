# Copyright ©2022-2023. The Regents of the University of California (Regents). 
# All Rights Reserved. 

# Permission to use, copy, modify, and distribute this software and its 
# documentation for educational, research, and not-for-profit purposes, without
# fee and without a signed licensing agreement, is hereby granted, provided that 
# the above copyright notice, this paragraph and the following two paragraphs 
# appear in all copies, modifications, and distributions. Contact The Office of
# Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, 
# CA 94720-1620, (510) 643-7201, otl@berkeley.edu, 
# http://ipira.berkeley.edu/industry-info for commercial licensing 
# opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
# OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS 
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED 
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE 
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

from pyquadkey2.quadkey import QuadKey  # type: ignore[import]
from shapely.geometry import Polygon  # type: ignore[import]


def quadkey_to_polygon(x: str) -> Polygon:
    """
    Generate polygons corresponding to 'quadkeys' from Microsoft Bing Maps Tile System
    https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system

    Args:
        x: quadkey string

    Returns: shapely polygon object
    """
    qk = QuadKey(x)

    # Get lat/lon pairs of four corners
    coords = [qk.to_geo(anchor=i) for i in range(4)]
    coords[2], coords[3] = coords[3], coords[2]
    coords = [(x[1], x[0]) for x in coords]

    # Create polygon from points and return
    return Polygon(coords)
