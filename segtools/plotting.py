import numpy as np
import matplotlib.pyplot as plt

from . import patchmaker
from . import lib

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
  
  coords = np.indices((4,5))*30
  plotimg = patchmaker.piece_together_ragged_2d(patches, coords.reshape(2,-1).T)
  plt.imshow(plotimg)
