from helpers import *
from geokit.util import *

## Scale Matrix
def test_scaleMatrix():
  # setup
  sumCheck = MASK_DATA.sum()

  # Equal down scale
  scaledMatrix1 = scaleMatrix(MASK_DATA,-2)
  
  if(scaledMatrix1.sum()*2*2 != sumCheck): error("scaledMatrix 1")

  # Unequal down scale
  scaledMatrix2 = scaleMatrix(MASK_DATA,(-2,-4))
  
  if(scaledMatrix2.sum()*2*4 != sumCheck): error("scaledMatrix 2")

  # Unequal up scale
  scaledMatrix3 = scaleMatrix(MASK_DATA,(2,4))
  
  if(scaledMatrix3.sum()/2/4 != sumCheck): error("scaledMatrix 3")

  # Strict downscale fail
  try:
    scaledMatrix4 = scaleMatrix(MASK_DATA,-3)
    error("scalingMatrix 4 - strict scaling fail")
  except GeoKitError as e:
    pass
  else:
    error("scalingMatrix 4 - strict scaling fail")

  # non-stricr downscale
  scaledMatrix5 = scaleMatrix(MASK_DATA,-3, strict=False)
  #print(scaledMatrix1.sum()*3, sumCheck)

if __name__=="__main__":
  test_scaleMatrix()