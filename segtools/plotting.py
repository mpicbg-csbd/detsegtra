import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.spatial import Voronoi

from skimage.segmentation import find_boundaries


from . import patchmaker
# from . import nhl_tools
from . import scores_dense as ss
from . import color
from . import label_tools


def nuc_grid_plot(img, nhl):
  if len(nhl) > 100:
      print("Too many nuclei! Try again w len(nhl) < 100.")
      return False

  def f(i):
      img_crop = lib.nuc2img(nhl[i], img, 4)
      lab, ncells = lib.label(img_crop > 0.92)
      lab = lab.sum(2)
      return lab
  patches = [f(i) for i in range(len(nhl))]

  coords = np.indices((4, 5))*30
  plotimg = patchmaker.piece_together_ragged_2d(
      patches, coords.reshape(2, -1).T)
  plt.imshow(plotimg)

def nhl2crops(img, nhl, axis=None, pad=10):
    def f(i):
        ss = lib.nuc2slices(nhl[i], pad, shift=pad)
        img_crop = img[ss].copy()
        if img_crop.ndim==3 and axis is not None: 
            print('hooligans', ss)
            a = img_crop.shape[axis]
            img_crop = img_crop[a//2]
        return img_crop
    patches = [f(i) for i in range(len(nhl))]
    return patches

def plot_nhls(nhls,
                x=lambda n:n['coords'][0], 
                y=lambda n:np.log2(n['area'])):
    cm = sns.cubehelix_palette(len(nhls))
    for i,nhl in enumerate(nhls):
        xs = [x(n) for n in nhl]
        ys = [y(n) for n in nhl]
        plt.scatter(xs, ys, c=cm[i])

def ax_scatter_data(ax, data, **kwargs):
    """
    matplotlib scatterplot, but you can use a list of dicts
    most common labels are plotted first, (underneath).
    """
    xs = np.array([d['x'] for d in data])
    ys = np.array([d['y'] for d in data])
    cs = np.array([d['c'] for d in data])
    szs = np.array([d['s'] for d in data])
    labels = np.array([d['l'] for d in data])

    
    print(kwargs)

    # sort labels from most to least frequent
    labelset, labelcts = np.unique(labels, return_counts=True)
    inds = np.argsort(labelcts)

    for l in labelset[inds][::-1]:
        mask = labels == l
        ax.scatter(xs[mask], ys[mask], s=szs[mask], c=cs[mask], label=l, **kwargs)

def ax_scatter_plus(ax, xs, ys, cs, labels, szs, **kwargs):
    """
    matplotlib scatterplot, but you can use a list of labels
    most common labels are plotted first, (underneath).
    """
    data = {}
    data['x'] = xs
    data['y'] = ys
    data['c'] = cs
    data['s'] = szs
    data['l'] = labels
    ax_scatter_data(ax, data, **kwargs)

def lineplot(img):
    pal = sns.diverging_palette(255, 133, l=60, n=7, center="dark")
    sns.set_palette(pal)
    a, b, c = img.shape
    lines = img[a//2, ::100].reshape(-1, c)
    fig = plt.figure()
    ax = fig.gca()
    for l in lines:
        ax.plot(l)
    ax2 = fig.add_axes([0.23, 0.50, 0.3, 0.3])
    ax2.imshow(img.max(0))

def make_comparison_image(img_raw, lab, lab_gt,
                          matching,
                          ax=None,
                          col_tp = (0,1,0,1),
                          col_fp = (1,0,0,1),
                          col_fn = (1., 0.41, 0.71,1),
                          **kwargs):
  if ax is None:
    ax = plt.gca()

  # compute matching and unmatching cell masks
  # m1, m1c matching and unmatching ground truth
  # m2, m2c matching and unmatching predictions 
  matchstuff = ss.sets_maps_masks_from_matching(lab_gt, lab, matching)
  m1, m2, m1c, m2c = matchstuff['masks']
  map1, map2 = matchstuff['maps']
  labcopy = color.recolor_from_mapping(lab, map2)
  if m2c.sum() > 0:
    lab[m2c] = lab[m2c] - lab[m2c].min() + lab_gt[m1].max() + 1
  lab[m2] = labcopy[m2]

  res = ax.imshow(img_raw, cmap=plt.cm.Greys_r, **kwargs)

  trans= (0,0,0,0)
  img = np.zeros(lab.shape, np.uint16)

  img[m2]  = 1
  img[m2c]  = 2
  img[m1c]  = 3

  cmap = np.array([trans, col_tp, col_fp, col_fp],np.float32)
  img  = cmap[img.flat].reshape(img_raw.shape + (4,))
  img[...,-1] *= .2
  
  res = ax.imshow(img)
    # relabel and color
  borders_lab = find_boundaries(lab, mode = "thick")
  borders_gt = find_boundaries(lab_gt)

  
  img = np.zeros(lab.shape, np.uint16)

  # draw borders 
  img[~borders_gt] = 0
  img[borders_gt] = 0
  # unmatched false negatives
  img[borders_gt & m1c] = 4
  #matched
  img[borders_lab & m2] = 2
  # unmatched false positives
  img[borders_lab & m2c] = 3

  trans= (0,0,0,0)

  cmap = np.array([trans, trans,col_tp, col_fp, col_fn], dtype=np.float32)

  img  = cmap[img.flatten()].reshape(img_raw.shape + (4,))
  img[...,-1] *= .7
  
  res = ax.imshow(img)



  return ax

def make_comparison_image_input(img_raw,  lab_gt, ax=None,
                                col_gt = (1., 0.41, 0.71,1),
                                **kwargs):

  if ax is None:
    ax = plt.gca()
  
  # relabel and color

  ggt = lab_gt.copy()
    
  border = find_boundaries(lab_gt, mode = "outer")

  
  trans      = (0,0,0,0.0)

  img = np.zeros(lab_gt.shape, np.uint16)

  img[~border] = 0
  img[border] = 1

  cmap = np.array([trans, col_gt], dtype=np.float32)

  img  = cmap[img.flat].reshape(img_raw.shape + (4,))
  img[...,-1] *= .7

  
  res = ax.imshow(img_raw, cmap=plt.cm.Greys_r, **kwargs)
  res = ax.imshow(img)


  return ax

def voronoi_plot_2d_w_colors(points, colors, **kwargs):
    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    for i,region in enumerate(regions):
        polygon = vertices[region]
        # rgb = np.random.rand(3)
        if hasattr(colors[i],'__len__'):
          rgb = colors[i]
        else:
          rgb = (float(colors[i]),)*3 / colors.max()

        x = polygon[:,0]
        y = polygon[:,1]
        plt.fill(x,y,c=rgb,alpha=0.4)

    plt.scatter(points[:,0], points[:,1], marker='o', c='k', **kwargs)
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Taken from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint  = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def fig2data(fig, shape):
  "https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array"
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

  canvas = FigureCanvas(fig)
  canvas.draw()       # draw the canvas, cache the renderer
  
  ax = fig.gca()
  ax.axis('off')
  
  fig.set_dpi(100)
  fig.set_size_inches(shape[1]/100, shape[0]/100, forward=True)
  fig.tight_layout(pad=0)
  width, height = shape #fig.get_size_inches() * fig.get_dpi()

  image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
  image = image.reshape((int(height), int(width), 3))
  return image
