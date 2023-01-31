# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: 

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the 
# documentation and/or other materials provided with the
# distribution. 

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
