import os

scripts = ["util", "rasterUtil", "vectorUtil", "extent", "regionMask", "algorithms", "indicators", "exclusionCalculator"]

for script in scripts:
    print("Testing %s..."%script)
    res = os.system("python test.%s.py"%script)
    
    if(res!=0):
        break
    else:
        print("")

print("Yay!")