from .defaults.ipython import watershed, distance_transform_edt, label, np
# from scipy.optimize import minimize, basinhopping, differential_evolution
# from scipy.optimize import OptimizeResult


def watershed_two_chan(x, nuc_mask=0.5, nuc_seed=0.8, mem_mask=1.0, mem_seed=1.0, ch_nuc=1, ch_mem=0, **kwargs):
  x1 = x[...,ch_nuc] # nuclei
  x2 = x[...,ch_mem] # borders
  poten = 1 - x1 #- x2
  mask = (x1 > nuc_mask) & (x2 < mem_mask)
  seed = (x1 > nuc_seed) & (x2 < mem_seed)
  res  = watershed(poten, label(seed)[0], mask=mask, **kwargs)
  return res

def watershed_memdist(x, nuc_mask=0.5, nuc_seed=0.8, mem_mask=1.0, dist_cut=5.0, ch_nuc=1, ch_mem=0, **kwargs):
  x1 = x[...,ch_nuc] # nuclei
  x2 = x[...,ch_mem] # borders
  poten = 1 - x1 #- x2
  membrane_mask = x2 > mem_mask
  dist = ~membrane_mask
  dist = distance_transform_edt(dist, sampling=[5,1,1])
  mask = (x1 > nuc_mask) #& (x2 < mem_mask)
  seed = (x1 > nuc_seed) & (dist > dist_cut)
  res  = watershed(poten, label(seed)[0], mask=mask, **kwargs)
  return res    

def flat_thresh_two_chan(x, nuc_mask=0.5, mem_mask=0.5, ch_nuc=1, ch_mem=0):
  x1 = x[..., ch_nuc] # nuclei
  x2 = x[..., ch_mem] # borders
  mask = (x1 > nuc_mask) & (x2 < mem_mask)
  res = label(mask)[0]
  return res

## --- objective functions ---

# def noop_min(fun, x0, args, **options):
#   return OptimizeResult(x=x0, fun=fun(x0, *args), success=True, nfev=1)
  

def poisson_disk_sample(width=1.0, height=1.0, radius=0.25, k=30):
    "https://pythonexample.com/code/poisson%%20disk%%20sampling/"
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
 
    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        P = np.empty((k, 2))
        P[:, 0] = p[0]+R*np.sin(T)
        P[:, 1] = p[1]+R*np.cos(T)
        return P
 
    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < height
 
    def neighborhood(shape, index, n=2):
        row, col = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        I = np.dstack(np.mgrid[row0:row1, col0:col1])
        I = I.reshape(I.size//2, 2).tolist()
        I.remove([row, col])
        return I
 
    def in_neighborhood(p):
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        if M[i, j]:
            return True
        for (i, j) in N[(i, j)]:
            if M[i, j] and squared_distance(p, P[i, j]) < squared_radius:
                return True
        return False
 
    def add_point(p):
        points.append(p)
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        P[i, j], M[i, j] = p, True
 
    # Here `2` corresponds to the number of dimension
    cellsize = radius/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
 
    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius
 
    # Positions cells
    P = np.zeros((rows, cols, 2), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)
 
    # Cache generation for neighborhood
    N = {}
    for i in range(rows):
        for j in range(cols):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)
 
    points = []
    add_point((np.random.uniform(width), np.random.uniform(height)))
    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[M]