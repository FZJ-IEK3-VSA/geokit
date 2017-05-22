from helpers import *

def placeItemsInMatrix_():
	## Item placement
	matrix_locs1 = placeItemsInMatrix(ELIGIBILITY_DATA, distance=3)
	if( len(matrix_locs1[0]) != 647 ): error("Item placement 1")

	matrix_locs2 = placeItemsInMatrix(ELIGIBILITY_DATA, distance=3, fastMethod=True)
	if( len(matrix_locs2[0]) != 667 ): error("Item placement 2")

def placeItemsInRaster_():
	## Place in Raster
	piir1 = placeItemsInRaster(AACHEN_ELIGIBILITY_RASTER, distance=0.01)#, output=result("algorithms_raster_locs_1.shp"))

	if not (piir1.GetGeometryCount()==527): error("placeItemsInRaster 1 - item count")

	# place items in raster using the fast method and a non-default placement division
	piir2 = placeItemsInRaster(AACHEN_ELIGIBILITY_RASTER, distance=0.01, fastMethod=True, placementDiv=20) #, output=result("algorithms_raster_locs_2.shp"))

	if not (piir2.GetGeometryCount()==541): error("placeItemsInRaster 2 - item count")

	# output as exclusion circles and write to file
	piir3 = result("algorithms_raster_locs_3.shp") # just make a name for the output file
	placeItemsInRaster(AACHEN_ELIGIBILITY_RASTER, distance=0.005, placementDiv=20, output=piir3, outputAsPoints=False, overwrite=True)

	res3 = list(vectorItems(piir3))

	if not (len(res3)==1755): error("placeItemsInRaster 3 - item count")
	if not (res3[10][0].GetGeometryName()=="POLYGON"): error("placeItemsInRaster 3 - item type")

	exc = 2*np.sqrt(res3[10][0].Area()/np.pi)
	if not abs(1-res3[10][1]["exclusion"]/exc)<0.005: error("placeItemsInRaster 3 - exclusion diameter")

def growMatrix_():
	print("growMatrix needs to be expanded...")
	## Grow matricies
	grown1 = growMatrix(MASK_DATA, dist=5, div=20)
	if(grown1.sum()!=4459.63): error("Grow matrix 1")

def overlayRasters_():
	## Overlay rasters
	overlayRasters(result("algorithms_overlayRasters.tif"), "data/divided_raster_*.tif", overwrite=True)

def combineRasters_():
	print("MAKE A TESTING SCRIPT FOR COMBINERASTERS!!!!")

def coordinateFilter_():
	print("MAKE A TESTING SCRIPT FOR COORDINATEFILTER!!!!")

if __name__ == "__main__":
	placeItemsInMatrix_()
	placeItemsInRaster_()
	growMatrix_()
	combineRasters_()
	coordinateFilter_()