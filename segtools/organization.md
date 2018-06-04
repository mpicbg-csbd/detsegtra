*What's the logic behind the modules we've created?*

# core libraries

## stack_segmentation.py
We need a library of 3D stack segmentation methods
- should 2d methods go here?
- what about resegmentation of individual oversegments?
- these should be instance segmentation heuristics.
- gmm fitting is a core method and should be included!
    + both fitting to masks and to intensities.

## unet.py
functions to create tensorflow models
- unet model
- batch generators?
    + do we gain anything by factoring this out of data script into unet.py?
    + we have to pass functions as arguments to generators so we can augment, etc, from script.

## ??
functions to generate training data / work with data
- normalization (all data-specific & done in scripts)

## lib.py
functions for turning segmented images (hyp) into nhl's.
- hyp2nhl
- should the re-segmentation & gmm fitting tools go here?
- should we rename nuclear-hypothesis-list into object-hypothesis-list?
- lib to nhl_tools? ohl_tools?

## track_tools.py
functions for turning nhls into tracking solutions
- recoloring images based on lineage
- plotting arrows for flow on top of stacks
- needed params are:
    + k and dub for graph_connections
    + edge,velgrad costs (scale? offset? nonlinearity?)
    + cutoff dist for velgrad neighbors. or cost decay w distance? physically motivated?
    + appearance costs? (size difference, etc)
+ 

## numpy_utils.py
functions for reshaping and permuting numpy arrays
- perm, merg, splt, collapse
- apply functions/operations across subset of dimensions

## patchmaker.py
converting images into patches and back again
- grid sample patches
- random sample patches
- sewing patches back together
- applying functions to patches / block-wise
- should this be combined with [numpy_utils.py]() ? 

## python_utils.py
simple python functions that should always be available

## scores_dense.py
tools for scoring segmentations
- work with matchings between label images (dense object representation)
- should there be an equivalent matching & scoring sparse object representation?
    + There is! [graphmatch.py]()

## graphmatch.py
tools for matching and scoring sparse object representations
- (symmetric) (k) nearest neighbor matching
- connect points (symmetric, digraph, digraph_symmetric)

## label_tools.py
tools for working with label images. masking and boundaries.
- identify boundaries
- neighbor dists
- maybe move pixel_bipartite() here?
- masking tools

## math_utils.py
coordinate transforms, integer factorization

## augmentation.py
functions for performing data augmentation
- warping?
- noise and blur should always be data-specific? like normalization? should live in script?

## ipython_bash.py
work with files and directories in standard project form
- compare series of diffs
- plot histories/results from multiple directories
- run function on all images in directory and resave
- 


# visualization library

## plotting.py
rendering segmentations, plotting segmentation overlays

## colors.py
work with colors
- recolor label images
- generate random colormaps of various kinds
- how much does seaborn automate this for us?

## rendering
turn 3D stacks into nice 2d images
- color by z-index of max
- only orthogonal projections?
- show first non-zero label

# interactive stuff

## spima.py
controling spimagine, curating lists of hypotheses
- moveit
- curate
- render movie
- save rgb still img

## cell_view_lib.py (replaced by stackviewer?)
stackviewer (not in segtools. should it be?)
we have cell_view_lib, but it's largely irrelevant...
- interact with scatterplots? (should belong to spima?)
- add listeners to stackviewer? (should belong to stackviewer or project-dependent.)


# dependency tree factorizations

ipython_local_defaults.py
ipython_defaults.py
ipython_remote_defaults.py
train_defaults.py

These are ways of keeping the dependencies between scripts consistent.
These are for making it quick and easy to start ipython and have a reasonable working env.


# needed?

library for working with annotations
- annotations come in many forms
    + lists of object annotations from spima.curate
    + dense annotations from pixelwise drawings
    + these two forms of annotations match up with our two representations of objects!
    + dense and sparse!
    + dense annotations can also be simple pixelwise classes, with no implied instance segmentation
- what functions do we need?
    + re-segment given correct annotations
        * run arbitrary code on an image slice given an anotation.
    + note: the precise sparse-annotation object-classes are project dependent.
    + maybe this isn't enough to justify it's own module...?