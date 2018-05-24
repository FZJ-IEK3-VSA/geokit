from helpers import *
from geokit.srs import *

def test_xyTransform():

    # test single point
    p1 = xyTransform( pointInAachen3035, fromSRS='europe_m', toSRS='latlon')[0]
    real = (6.313298792067333, 50.905105969570265)
    if not (isclose(p1[0],real[0], 1e-6) and isclose(p1[1],real[1], 1e-6)):
        error("xyTransform 1")

    # test multiple points
    p2 = xyTransform( pointsInAachen4326, fromSRS='latlon', toSRS='europe_m')
    real = [(4042131.1581, 3052769.5385), (4039553.1900, 3063551.9478), (4065568.415, 3087947.743)]
    for p, r in zip(p2,real):
        if not (isclose(p[0],r[0], 1e-6) and isclose(p[1],r[1], 1e-6)):
            error("xyTransform 1")

    print("xyTransform passed all tests")

def test_loadSRS():
    # Test creating an osr SRS object
    s1 = loadSRS(EPSG4326)
    
    # Test an EPSG identifier
    s2 = loadSRS(4326)

    # Are they the same?
    if not s1.IsSame(s2): error("SRS mismatch")

    print( "loadSRS passed all tests") 

if __name__ == '__main__':
    test_xyTransform()
    test_loadSRS()