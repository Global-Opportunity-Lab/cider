from pyquadkey2.quadkey import QuadKey
from shapely.geometry import Polygon


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
