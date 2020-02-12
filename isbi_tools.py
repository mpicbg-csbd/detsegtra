import numpy as np
from skimage.measure import regionprops

def pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

