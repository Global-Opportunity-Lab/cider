from pyquadkey2.quadkey import QuadKey
from shapely.geometry import Polygon
from pandas import Series


def quadkey_to_polygon(x: Series) -> Polygon:
    qk = QuadKey(x['quadkey'])
    # Get lat/lon pairs of four corners
    coords = [qk.to_geo(anchor=i) for i in range(4)]
    coords[2], coords[3] = coords[3], coords[2]
    coords = [(x[1], x[0]) for x in coords]
    # Create polygon from points and return
    return Polygon(coords)
