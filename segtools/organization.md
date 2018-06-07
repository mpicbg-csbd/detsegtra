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
- needed params are:
    + k and dub for graph_connections
        * knn_t+1, dub_t+1
    + edge,velgrad costs (scale? offset? nonlinearity?)
        * edge_cost_scale, velgrad_cost_scale
    + cutoff dist for velgrad neighbors. or cost decay w distance? physically motivated?
        * neib_graph_cutoff
    + appearance costs? (size difference, etc)
if we use **kwargs in definition, then we don't get to capture any names in current scope; we only pass them on.
if we have some unused keyword args passed into any func in the calltree without a **kwargs in func def then errors.
the callee defines names to which the caller must adhere. or at least the names must be consistent.
positional args allows for renaming values at function boundaries.
you can still rename values with keyword args, but the caller must know the name of the callee.
if you pass a dictionary then the dictionary can be renamed, but the vals inside cannot.
the caller and callee must agree on dict internals.
in a tall call tree we must pass the dict around as an arg throughout the whole tree, even if it never changes inside any func body.
an alternative to passing around params through deep call trees is to have a global name that the funcs can reference. 
We can either define a global name that lives as a module attribue (same level as function names), or we can encapsulate the functions and these global names inside a class.
Encapsulation inside a class enforces that we can't call the functions until we have instantiated the class.
it also allows us to have mutliple instantiations / tracking problems setup in the same python instance with different params, while still mainting the ease of global state.
the obnoxious part of classes is the separate "initialize" and "call" steps.
can you use kwargs like a normal dictionry of params if you want to avoid providing a default param value?
This means you have to reference your variable name inside of strings instead of giving it a proper name in the scope.
python allows you to pass around dictionaries.
if i want to enfore shared names between caller and callee.
it matters that we know who the callers are in advance.
so grouping code into functions is only to abstract procedure names, not argument names.
even positional args names are usually shared between caller and callee throughout the library.

## numpy_utils.py
functions for reshaping and permuting numpy arrays
- perm, merg, splt, collapse
- apply functions/operations across subset of dimensions

## patchmaker.py
converting images into patches and back again
- we should rethink how this works. the main function should probably be to generate lists/ndarrays of slices.
- this generalizes to 3D, allows for ragged shapes easily, and can be used to re-assign after transformations as long as shape doesn't shange.
- takes up less memory than copying entire stacks. usually we don't want to copy.
- padding/boundary conditions are the responsibility of the caller!
- using an overlap when building slice list is easier than stride tricks directly on ndarray!
- applying functions to patches and re-sowing becomes trivial. What about when patch size changes?
- 
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

## track_vis.py
- plotting circles and arrows for flow on top of stacks
- integration with stackviewer and spimagine for viewing and editing tracking solns?
- 

# interactive stuff
## annotation.py
- centerpoint annotation tool
- Lasso selectors that interact with spimagine
- lasso selectors that interact with Stack
- lasso selectors to draw polygons on to Stack
- potentially merge with cell_view_lib? with spima?
- 

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