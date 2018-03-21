# Should the API be based off of nx graphs? Should it require nhl or hyp?

To compute seg we need the original images to determine the pixel overlap. But most of our other matchings would be based off of nhls.

On the other hand we need to be able to compute seg given pairs of images, because that's a sensible interface... But there are many statistics that we can only compute given a full-on matching matrix...

- If the nhl's knew their exact pixels, then we'd be able to use them for calculating pixel overlap.
- For many other graphs we want to build connections between OBJECTS. not just a direct comparison at the pixel level. Maybe this would actually be a better loss function? Comparison at the object level...
- We could keep the image-level comparisons separate from the object-level ones... An object comparison would take a pair of nhls. An image-level one would take a pair of hyps or labs. An image level comparison could be RAND, WARP, SEG, etc. It could use ndarray bipartite graphs. Both could return scores, potentially recolorings/relabelings, matchings, etc. But for image-level comparisons we would use the pixels for determining the matching. Everything goes through pixel_sharing_bipartite. And for object level everything goes through the data in nhl. But the stuff they return is the same? Do all object level comparisons use a graph library?

# Should we factor out the code that deps on gputools into a separate lib?

- full anaconda dist
    + numpy
    + matplotlib
    + numba
    + pyopencl
    + networkx
    + seaborn
    + scipy
    + skimage
    + sklearn
- gputools
- spimagine
- pykdtree

- Our complex deps are matplotlib (specific Qt backend), gputools (requires gpu?), spimagine (requires gputools), pykdtree and numba. At least pyopencl, numba, networkx and matplotlib all come by default with the anaconda dist. But we'll have to install pykdtree separately!

- keras
- tensorflow

- We could potentially include warping and unet libs to put everything in one library! But then we need to include keras & tensorflow backends as well.

# where does label border detection belong?

It's currently in voronoi, but it could be in segtools_simple (which should probably be renamed...). The cool plots from the stardist paper use segtools_simple for getting matching masks/sets/maps and then get object borders from voronoi.

# Should voronoi be segmentation? How to incorporate ndtess? pathfinder?

- Can we turn pathfinder into a numba program?
- Is it simple to build ndtess and incorporate pybind11?

# What scores to add to segtools_simple?

- RAND
- Warping error
- Tolerant Edit Distance

