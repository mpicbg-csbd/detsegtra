import gputools

def spectral_clustering_seg(img, n_clusters=8, threshold=150):
  """
  uses spectral clustering on pixel graph to split connected components.
  only included here for posterity
  shitty method for nuclei. also very slow. img is uint16? HypothesisImage.
  """
  from sklearn.feature_extraction.image import img_to_graph
  from sklearn.cluster import spectral_clustering
  bimg = gputools.blur(img)
  plt.imshow(bimg)
  mask=bimg>threshold
  graph = img_to_graph(bimg, mask=mask)
  graph.data = np.exp(-graph.data/graph.data.std())
  labels = spectral_clustering(graph, n_clusters=8)
  labelsim = np.zeros(bimg.shape)
  labelsim[mask] = labels
  return labelsim


def mask2cmask(mask, dil_iter=2, sig=3.0):
  mask = nd.binary_dilation(mask, iterations=2).astype('float')
  hx = gaussian(9, 3.0)
  mask = gputools.convolve_sep3(mask, hx, hx, hx)
  mask = mask/mask.max()
  mask = 1-mask
  return mask
