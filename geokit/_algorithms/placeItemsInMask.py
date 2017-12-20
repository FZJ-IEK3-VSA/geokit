from geokit._core.regionmask import *

def placeItemsInMask(mask, separation, extent=None, pixelDivision=5, maxItems=10000000, outputSRS=None):
        """Distribute the maximal number of minimally separated items within the available areas
        
        Returns a list of x/y coordinates (in the ExclusionCalculator's srs) of each placed item

        Inputs:
            separation - float : The minimal distance between two items

            pixelDivision - int : The inter-pixel fidelity to use when deciding where items can be placed

            preprocessor : A preprocessing function to convert the accessibility matrix to boolean values
                - lambda function
                - function handle

            maxItems - int : The maximal number of items to place in the area
                * Used to initialize a placement list and prevent using too much memory when the number of placements gets absurd
        """
        # Preprocess availability
        if isinstance(mask,str):
            mask = extractMatrix(mask)
        if not mask.dtype == "bool":
            raise RuntimeError("Mask must be a bool type")
        workingAvailability = preprocessor(s.availability)
        if not workingAvailability.dtype == 'bool':
            raise s.GlaesError("Working availability must be boolean type")
        workingAvailability[~s.region.mask] = False
        # Turn separation into pixel distances
        separation = separation / s.region.pixelSize
        sep2 = separation**2
        sepFloor = max(separation-1,0)
        sepFloor2 = sepFloor**2
        sepCeil = separation+1

        # Make geom list
        x = np.zeros((maxItems))
        y = np.zeros((maxItems))

        bot = 0
        cnt = 0

        # start searching
        yN, xN = workingAvailability.shape
        substeps = np.linspace(-0.5, 0.5, pixelDivision)
        substeps[0]+=0.0001 # add a tiny bit to the left/top edge (so that the point is definitely in the right pixel)
        substeps[-1]-=0.0001 # subtract a tiny bit to the right/bottom edge for the same reason
        
        for yi in range(yN):
            # update the "bottom" value
            tooFarBehind = yi-y[bot:cnt] > sepCeil # find only those values which have a y-component greater than the separation distance
            if tooFarBehind.size>0: 
                bot += np.argmin(tooFarBehind) # since tooFarBehind is boolean, argmin should get the first index where it is false

            #print("yi:", yi, "   BOT:", bot, "   COUNT:",cnt)

            for xi in np.argwhere(workingAvailability[yi,:]):
                # Clip the total placement arrays
                xClip = x[bot:cnt]
                yClip = y[bot:cnt]

                # calculate distances
                xDist = np.abs(xClip-xi)
                yDist = np.abs(yClip-yi)

                # Get the indicies in the possible range
                possiblyInRange = np.argwhere( xDist <= sepCeil ) # all y values should already be within the sepCeil 

                # only continue if there are no points in the immediate range of the whole pixel
                immidiateRange = (xDist[possiblyInRange]*xDist[possiblyInRange]) + (yDist[possiblyInRange]*yDist[possiblyInRange]) <= sepFloor2
                if immidiateRange.any(): continue

                # Start searching in the 'sub pixel'
                found = False
                for xsp in substeps+xi:
                    xSubDist = np.abs(xClip[possiblyInRange]-xsp)
                    for ysp in substeps+yi:
                        ySubDist = np.abs(yClip[possiblyInRange]-ysp)

                        # Test if any points in the range are overlapping
                        overlapping = (xSubDist*xSubDist + ySubDist*ySubDist) <= sep2
                        if not overlapping.any():
                            found = True
                            break

                    if found: break

                # Add if found
                if found:
                    x[cnt] = xsp
                    y[cnt] = ysp
                    cnt += 1
                 
        # Convert identified points back into the region's coordinates
        coords = np.zeros((cnt,2))
        coords[:,0] = s.region.extent.xMin + (x[:cnt]+0.5)*s.region.pixelWidth # shifted by 0.5 so that index corresponds to the center of the pixel
        coords[:,1] = s.region.extent.yMax - (y[:cnt]+0.5)*s.region.pixelHeight # shifted by 0.5 so that index corresponds to the center of the pixel

        # Done!
        s.itemCoords = coords

        if outputSRS is None:
            return coords
        else:
            newCoords = gk.srs.xyTransform(coords, fromSRS=s.region.srs, toSRS=outputSRS)
            newCoords = np.column_stack( [ [v[0] for v in newCoords], [v[1] for v in newCoords]] )
            return newCoords

