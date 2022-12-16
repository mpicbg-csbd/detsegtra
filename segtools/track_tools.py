from tabulate import tabulate
import numpy as np
import networkx as nx
import pulp
from collections import defaultdict, Counter
from sortedcollections import SortedDict
from PIL import Image, ImageDraw
# from multipledispatch import dispatch
import matplotlib.pyplot as plt
import scipy.spatial as sp
from numba import jit

import ipdb

from . import scores_dense as ss
from . import nhl_tools
from . import graphmatch as gm
from . import color
from .python_utils import print_sorted_counter, reduce

## TrackFactory is a class so

class TrackFactory(object):
    """
    Provides methods for assignment tracking on nhls.
    Builds a networkx graph of potential connections across time.
    Build a PuLP problem with costs for each connection.
    Minimizes the resulting the LP problem
    """

    def __init__(self,
                knn_n            = 15,
                knn_dub          = 50,
                edgecost         = None,
                vertcost         = None,
                straincost       = None,
                allow_div        = False,
                allow_exit       = False,
                allow_appear     = False,
                neib_edge_cutoff = 70,
                on_edges         = None,
                ):

        graph_cost_stats = []
        self.knn_n              = knn_n
        self.knn_dub            = knn_dub
        self.neib_edge_cutoff   = neib_edge_cutoff
        self.graph_cost_stats   = graph_cost_stats
        # self.do_velcorr         = do_velcorr
        # self.velgrad_scale      = velgrad_scale
        self.on_edges           = on_edges
        self.edgecost           = edgecost
        self.vertcost           = vertcost
        self.straincost         = straincost
        self.allow_div          = allow_div
        self.allow_exit         = allow_exit
        self.allow_appear       = allow_appear

        # self.on_edges = [((1,93), (2,100)),
        #             ((3,119), (4,96)),
        #             ((3,134), (4,107))]

    def nhls2tracking(self, nhls):
        graph = self._nhls2graph(nhls)
        nucdict = nhl_tools.nhls2nucdict(nhls)

        prob, vv, ev = self._graph2pulp(graph, nhls)
        # ipdb.set_trace()

        ## Debugging
        print("Status: ", prob.status)
        print("Vert Vars:", Counter([v.value() for v in vv.values()]))
        print("Edge Vars:", Counter([v.value() for v in ev.values()]))

        tv, te, tb = vars2tb(vv, ev)
        if tv[0]==[]:
            raise ValueError("No cells in the solution.")

        al = arrowlist(tb, tv, nucdict)
        cm = lineagecolormaps(tb, tv)

        tr = Tracking(graph, tb, tv, te, al, cm)
        return tr

    ## --- nhls2tracking depends on below

    def _nhls2graph(self, nhls):
        def mat(t): return np.array([n['centroid'] for n in nhls[t]])
        def labels(t): return [n['label'] for n in nhls[t]]
        bips = [gm.connect_points_digraph(mat(i),
                                          mat(i+1),
                                          lx=i,
                                          ly=i+1,
                                          labels_x=labels(i),
                                          labels_y=labels(i+1),
                                          k=self.knn_n,
                                          dub=self.knn_dub) for i in range(len(nhls)-1)]
        graph = reduce(nx.compose, bips[1:], bips[0])
        return graph

    def _graph2pulp(self, graph, nhls):
        prob = pulp.LpProblem("Assignment Problem", pulp.LpMinimize)

        nuc_dict = nhl_tools.nhls2nucdict(nhls)

        ## vertex and edge variables
        vertvars = pulp.LpVariable.dicts('verts', graph.nodes, lowBound=0, upBound=1, cat=pulp.LpBinary)
        vertcostdict = {n:self.vertcost(nuc_dict[n]) for n in graph.nodes}
        vertterm = [vertcostdict[n]*vertvars[n] for n in graph.nodes]

        edgevars = pulp.LpVariable.dicts('edges', graph.edges, lowBound=0, upBound=1, cat=pulp.LpBinary)
        edgecostdict = {(n1,n2):self.edgecost(nuc_dict[n1], nuc_dict[n2]) for (n1,n2) in graph.edges}
        edgeterm = [edgecostdict[e]*edgevars[e] for e in graph.edges]

        print("VERT & EDGE COSTS")
        x = np.array(list(vertcostdict.values()))
        self.graph_cost_stats.append((x.mean(), x.min(), x.max(), x.std()))
        x = np.array(list(edgecostdict.values()))
        self.graph_cost_stats.append((x.mean(), x.min(), x.max(), x.std()))


        OUTMAX = 2 if self.allow_div else 1
        print("OUTMAX = {}".format(OUTMAX))
        OUTMIN = 0 if self.allow_exit else 1
        print("OUTMIN = {}".format(OUTMIN))

        ## vertex and edge constraints
        for n in graph.nodes:
            e_out = [edgevars[(n,v)] for v in graph[n]]
            if len(e_out) > 0:
                prob += pulp.lpSum(e_out) <= OUTMAX * vertvars[n], ''
                prob += pulp.lpSum(e_out) >= OUTMIN * vertvars[n], ''  ## include to prevent on-vars from disappearing
        for n in graph.nodes:
            e_in = [edgevars[(v,n)] for v in graph.pred[n]]
            if len(e_in) > 0:
                if self.allow_appear:
                    prob += pulp.lpSum(e_in)  <= vertvars[n], ''
                else:
                    prob += pulp.lpSum(e_in)  == vertvars[n], ''

        if self.on_edges:
            for e in on_edges:
                prob += edgevars[e] == 1, ''

        ## objective
        if self.straincost is not None:
            vcterm = self._add_velocity_correlation(prob, graph, nhls, edgevars)
            prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm) + pulp.lpSum(vcterm), "Objective"
        else:
            prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm), "Objective"

        prob.writeLP('tracker.lp')
        prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 1200), ('OptimalityTol', 1e-7)]))
        # prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 100), ('ResultFile','coins.sol'), ('OptimalityTol', 1e-2)]))
        # prob.solve(pulp.PULP_CBC_CMD(options=['-sec 300']))
        return prob, vertvars, edgevars

    ## --- 2nd level of dependence

    def _add_velocity_correlation(self, prob, graph, nhls, edgevars):
        ## velocity correlation variables
        neighbor_graphs = self._build_neighbor_graphs(nhls)
        nuc_dict = nhl_tools.nhls2nucdict(nhls)
        neighbor_vars = []
        for ng in neighbor_graphs[:-1]:
            for e in ng.edges:
                u0,v0 = e
                for u1 in graph[u0]:
                    for v1 in graph[v0]:
                        if u1!=v1:
                            neighbor_vars.append((u0,u1,v0,v1))
        neighbor_vars_pulp = pulp.LpVariable.dicts('velocitygrad', neighbor_vars, lowBound=0, upBound=1, cat=pulp.LpBinary)
        costs = self._velocity_grad_cost_list(nuc_dict, neighbor_vars)
        x = np.array(costs)
        self.graph_cost_stats.append((x.mean(), x.min(), x.max(), x.std()))
        neighbor_vars_term = [costs[i]*neighbor_vars_pulp[vp] for i,vp in enumerate(neighbor_vars)]

        ## velocity correlation constraints
        for vp in neighbor_vars:
            (u0,u1,v0,v1) = vp
            prob += edgevars[(u0,u1)] + edgevars[(v0,v1)] >= 2 * neighbor_vars_pulp[vp], ''
            prob += edgevars[(u0,u1)] + edgevars[(v0,v1)] <= 1 + neighbor_vars_pulp[vp], ''

        return neighbor_vars_term

    ## --- bottom level of dependence

    def _build_neighbor_graphs(self, nhls):
        nucdict = nhl_tools.nhls2nucdict(nhls)
        points = lambda i: np.array([n['centroid'] for n in nhls[i]])
        labels = lambda i: [(i, n['label']) for n in nhls[i]]
        def f(points, labels):
            vor = sp.Voronoi(points)
            g = nx.from_edgelist([(labels[x], labels[y]) for x,y in vor.ridge_dict.keys()])
            remove_long_edges(g, nucdict, self.neib_edge_cutoff)
            return g
        neighbor_graphs = [f(points(i), labels(i)) for i in range(len(nhls))]
        return neighbor_graphs

    def _velocity_grad_cost_list(self, nucdict, neighbor_vars):
        def f(vp):
            u0=nucdict[vp[0]]['centroid']
            u1=nucdict[vp[1]]['centroid']
            v0=nucdict[vp[2]]['centroid']
            v1=nucdict[vp[3]]['centroid']
            return self.straincost(u0, u1, v0, v1)
        res = np.array([f(vp) for vp in neighbor_vars])
        return res

def cost_stats_lines(graph_cost_stats):
    lines = []

    def f(string):
        # print(string)
        lines.append(string)

    header  = " {: <7s}"*4
    header  = header.format("Mean", "Min", "Max", "Std")
    numline = "{: .4f} "*4

    if False:
        f("\n\nVert Costs")
        f(header)
        for x in graph_cost_stats[::3]:
            f(numline.format(*x))

    f("\n\nEdge Costs")
    f(header)
    for x in graph_cost_stats[1::3]:
        f(numline.format(*x))

    f("\n\nVelGrad Costs")
    f(header)
    for x in graph_cost_stats[2::3]:
        f(numline.format(*x))

    return lines

def compose_trackings(trackfactory, tracklist, nhls):
    """
    glue independent, overlapping tracking solutions back together.
    TODO: assert that the resulting solution is still valid!
    """
    tvs = [[(0, v[1]) for v in tracklist[0].tv[0]]]
    for i,tr in enumerate(tracklist):
        tvs.append([(i+1, v[1]) for v in tr.tv[1]])
    def f(i,tr): return [((i,u[1]), (i+1,v[1])) for u,v in tr.te[0]]
    tes = [f(i,tr) for i,tr in enumerate(tracklist)]
    tb = true_branching(tvs,tes)
    cm = lineagecolormaps(tb, tvs)
    # nhls = nhl_tools.filter_nhls(nhls)
    nucdict = nhl_tools.nhls2nucdict(nhls)
    graph = trackfactory._nhls2graph(nhls)
    al = arrowlist(tb, tvs, nucdict)
    tr = Tracking(graph, tb, tvs, tes, al, cm)
    return tr

## general graph manipulation

def remove_long_edges(g, nucdict, cutoff):
    badedges = []
    for e in g.edges:
        v,u = e
        x1,x2 = np.array(nucdict[v]['centroid']), np.array(nucdict[u]['centroid'])
        dr = np.linalg.norm(x1-x2)
        if dr > cutoff:
            badedges.append(e)
    g.remove_edges_from(badedges)

## utilities and data munging

# @DeprecationWarning

@DeprecationWarning
def nhls2nhldicts(nhls):
    def f(lis):
        d = dict()
        for n in lis:
            d[n['label']] = n
        return d
    return [f(nhl) for nhl in nhls]

## tools to build the True Branching and Tracking solution from the PuLP result

from collections import namedtuple
TrueBranching = namedtuple('TrueBranching', ['tb', 'tv', 'te'])
Tracking = namedtuple('Tracking', ('graph', 'tb', 'tv', 'te', 'al', 'cm'))

def vars2tb(vv, ev):
    tv = true_verts(vv)
    te = true_edges(ev)
    tb = true_branching(tv, te)
    # g = TrueBranching(tv, te, tb)
    return tv, te, tb

def true_verts_from_edges(tb, te):
    tv = [[n for n in tb.nodes if n[0]==i] for i in range(len(te)+1)]
    return tv

def true_verts(vv):
    groups = defaultdict(list)
    for v in vv.keys():
        if vv[v].value()==1:
            groups[v[0]].append(v)
    return list(SortedDict(groups).values())

def true_edges(ev):
    groups = defaultdict(list)
    for e in ev.keys():
        if ev[e].value() == 1:
            groups[e[0][0]].append(e)
    return list(SortedDict(groups).values())

def true_branching(tv, te):
    flatten = lambda l: [item for sublist in l for item in sublist]
    tb = nx.from_edgelist(flatten(te), nx.DiGraph())
    tb.add_nodes_from(flatten(tv))
    return tb

def arrowlist(tb, tv, nuc_dict):
    "associated an arrow with each node"
    def f(n):
        s = tb.pred[n] # set of predecessors. could be empty.
        if len(s)==1:
            n0 = list(s)[0]
            return nuc_dict[n0]['centroid'], nuc_dict[n]['centroid']
        else:
            return None
    def g(i):
        arrs = [f(n) for n in tv[i] if f(n) is not None]
        arrs = np.array(arrs)
        return arrs
    return [g(i) for i in range(1,len(tv))]

def lineagecolormaps(tb, tv):
    cm = [{0:(0,0,0)} for _ in range(len(tv))]
    for nset in nx.weakly_connected_components(tb):
        r = (np.clip(np.random.rand(3)/2 + 0.5, 0, 1)*255).astype(np.uint8)
        r = np.clip(np.random.rand(3)/2 + 0.5, 0, 1)
        # r = (255*r).astype(np.uint8)
        nset = sorted(nset)

        n = nset[-1]
        if n[0] == len(tv)-1:
            cm[n[0]][n[1]] = r
        else:
            cm[n[0]][n[1]] = (0,0,1) #np.array((0,0,255), dtype=np.uint8)

        n = nset[0]
        if n[0] == 0:
            cm[n[0]][n[1]] = r
        else:
            cm[n[0]][n[1]] = (0,1,0) #np.array((0,255,0), dtype=np.uint8)

        for n in nset[1:-1]:
            cm[n[0]][n[1]] = r
    return cm

def lineagelabelmap(tb,tv):
    cm = [{0:0} for _ in range(len(tv))]
    for i, nset in enumerate(nx.weakly_connected_components(tb)):
        for n in nset:
            cm[n[0]][n[1]] = i
    return cm

def run_recursive_division_labeling(tb,tv):
    cm = [{0:0} for _ in range(len(tv))]
    current_label = 1
    minmaxparentdict = {}

    def f(current_node, current_label, parent):
        n = current_node
        cm[n[0]][n[1]] = current_label
        tup = minmaxparentdict.get(current_label, (n[0],n[0],parent))
        minmaxparentdict[current_label] = (tup[0],n[0],parent)
        print(n, current_label)
        children = tuple(tb[n])
        if len(children)==0:
            return current_label
        if len(children)==1:
            return f(children[0], current_label, parent)
        elif len(children)==2:
            cl1 = f(children[0],current_label+1, current_label)
            cl2 = f(children[1],cl1+1, current_label)
            return cl2
        else:
            print("ERROR THREE KIDS!")

    for i, nset in enumerate(nx.weakly_connected_components(tb)):
        start_node = sorted(list(nset))[0]
        current_label = f(start_node,current_label,0) + 1

    return cm,minmaxparentdict

def write_file_from_minmaxparentdict(mmpd):
    string = ""
    for k,v in mmpd.items():
        string += "{} {} {} {}\n".format(k,*v)
    return string


## compute statistics on a Tracking

def stats_tr(tr):
    return stats(tr.tv, tr.te, tr.tb)

def stats(tv, te, tb):
    # print("\nDescendants")
    # print_sorted_counter([len(nx.descendants(tb, n)) for n in tv[0]])

    # since descendants doesn't get all the chains it's better to find the length of weakly connected components
    print("\nWeakly Connected Components")
    print_sorted_counter([len(g) for g in nx.weakly_connected_components(tb)])

    print("Children over time")
    table = [["Time", "Nodes", "Out Edges", "Out Dist"]]
    for i in range(len(tv)-1):
        c = Counter([len(tb[v]) for v in tv[i]])
        table.append([i, len(tv[i]), len(te[i]), {0:c[0], 1:c[1], 2:c[2]}])
    i = len(tv)-1
    table.append([i, len(tv[i]), '---', '---'])
    print(tabulate(table))
    print("Totals")
    print('Nodes : ', sum([len(x) for x in tv]))
    print('Edges : ', sum([len(x) for x in te]))

    table = [["time"] + [str(n) for n in range(len(tv))]]
    for i in range(len(tv)):
        c = Counter([len(nx.ancestors(tb, n)) for n in tv[i]])
        table.append([i] + [c[j] for j in range(i+1)])
    print(tabulate(table))
    return table

def plot_flow_hist(tr):
    "note: for these plots we switch x&y s.t. up corresponds to x = image up"
    arrows = tr.al[0]
    delta = arrows[:,0] - arrows[:,1]
    plt.figure()
    plt.hist2d(-delta[:,1], delta[:,0])
    plt.figure()
    for i in range(delta.shape[0]):
        dy,dx = delta[i]
        plt.arrow(0,0,-dx,dy)
    xli,yli = delta[:,1].max(), delta[:,0].max()
    plt.xlim((-yli, yli))
    plt.ylim((-xli, xli))


## coloring and drawing

def ances_grouped(graph, n):
    groups = defaultdict(list)
    for obj in nx.ancestors(graph, n):
        groups[obj[0]].append(obj)
    return groups

def desce_grouped(graph, n):
    groups = defaultdict(list)
    for obj in nx.descendants(graph, n):
        groups[obj[0]].append(obj)
    return groups

def color_group(lab, group):
    mask = np.zeros(lab.shape, dtype=np.bool)
    for i in group.keys():
        s = {x[1] for x in group[i]}
        m = seglib.mask_labels(s, lab[i])
        mask[i] = m
    return mask

def relabel_every_frame(labs, cm):
    labr = []
    for i in range(len(labs)):
        maxval = max(max(cm[i].keys()), labs[i].max()) + 1
        cmap = np.zeros(maxval)
        for k,v in cm[i].items():
            cmap[k] = v
        labr.append(cmap[labs[i].flat].reshape(labs[i].shape))
    labr = np.array(labr)
    # if labr.shape[-1]==1: labr = labr[...,0]
    return labr

def recolor_every_frame(labs, cm):
    labr = []
    for i in range(len(labs)):
        cmap = np.zeros((max(max(cm[i].keys()), labs[i].max()) + 1, 3))
        for k,v in cm[i].items():
            cmap[k] = v
        labr.append(cmap[labs[i].flat].reshape(labs[i].shape + (3,)))
    labr = np.array(labr)
    if labr.shape[-1]==1: labr = labr[...,0]
    return labr


## draw arrows and flow

@DeprecationWarning
def drawarrows_pix(img, arrows):
    im = Image.fromarray((255*img).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    for i in range(len(arrows)):
        y0,x0 = arrows[i][0]
        y1,x1 = arrows[i][1]
        tup = ((x0,y0),(x1,y1))
        if img.ndim==3:
            fill = (255,255,255)
        else:
            fill = 255
        draw.line(tup, fill=fill, width=1)
    return np.asarray(im)

@DeprecationWarning
def draw_all_arrows(img, arrowlist):
    img2 = np.zeros_like(img)
    img2[0] = img[0]
    for i in range(len(arrowlist)):
        img2[i+1] = drawarrows_pix(img[i+1], arrowlist[i])
    return img2

## Utils

def writegraph(graph):
    lines = ["Id;Label;X;Y"]
    for v in graph.nodes:
        p = graph.nodes[v]
        lines.append('{};{};{};{}'.format(v,v,p['x'],p['y']))
    writelines('graph_nodes.csv', lines)
    lines = ["Source;Target"]
    for v in graph.edges:
        lines.append('{};{}'.format(v[0],v[1]))
    writelines('graph_edges.csv', lines)

def writelines(filename, lines):
    with open(filename, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(lines))

@DeprecationWarning
def set_track_time(tr, time):
    tb2.add_nodes_from([(v[0]+time,v[1]) for v in tr.tb.nodes])
    tv2 = [(v[0]+time,v[1]) for verts in tr.tv for v in verts]
    te2 = [((v[0]+time,v[1]), (u[0]+time, u[1])) for u,v in tr.te]
    tb2 = nx.from_edgelist([((v[0]+time,v[1]), (u[0]+time, u[1])) for u,v in tr.tb.edges], create_using=nx.DiGraph())
    return tb2



## isbi_submission_2021/tracking.py

"""
single cell tracking example from `https://napari.org/tutorials/applications/cell_tracking`
"""

import os
# import napari
# import ipdb
from pathlib import Path

import numpy as np
# import pandas as pd

from skimage.io import imread
from skimage.measure import regionprops, regionprops_table

from tifffile import imsave

import networkx as nx
import matplotlib
from matplotlib import pyplot as plt

# from segtools.ns2dir import load, save, toarray
# from segtools.math_utils import conv_at_pts_multikern
# from segtools.color import relabel_from_mapping
# from segtools import point_tools

import numpy_indexed as ndi
from pykdtree.kdtree import KDTree as pyKDTree
import re

from .place_label_spheres import place_label_spheres
from types import SimpleNamespace
from subprocess import run
from collections import defaultdict

# from time import time as pytime

COLUMNS = ['label', 'frame', 'centroid-0', 'centroid-1', 'centroid-2']


"""
Names in use:

Fixed Size
- dub "distance upper bound" the maximum distance allowed between matching points during nearest-neib tracking.
- scale "voxel scale" [Z]YX dimensions of image voxels. used to rescale points.
- info "ISBI dataset info" describes image shape, N time points, naming convention, etc
- kern "rasterized centerpoint kernel" a small ND-Array with a bright gaussian blob, used to rasterize pts into an image for DET/SEG/TRA evaluation.

Centerpoints
- [gt]pts "[ground truth] centerpoints" NxD arrays with D in [2,3].
- ltps "list of timepoints" a list of `[gt]pts`
- list_of_arrays "" list of 2D or 3D images

- lbep "Label Begin End Parent" The ISBI format used to describe tracking relationships. We use list of tuples?
- tb "True Branching" A NetworkX 'branching' that describes the relationships between objects over time. May have metadata describing object centerpoint.
- nap "Napari Tracking Format" Composed of `nap.tracklets,nap.graph,nap.properties`. Tracklets carry pts`, graph describes relationships.


---

Typical Usage

info = ...
ltps = [detect_pts(img) for img in list_of_images]
tb = nn_tracking_on_ltps(ltps=ltps, scale=(4,1,1), dub=20)
savedir = "./mydir/"
eval_tb_isbi(tb,info,savedir)

From this we get DET and TRA both!


"""


## Utility Funcs

def pad_and_stack_arrays(list_of_arrays, align_pt=None):
  """
  arrays don't need to be the same shape
  align_pt is list of pointers to array align_pt. to be aligned.
  TODO: make it work for arrays with channels, etc.
  """
  if align_pt is None: align_pt = np.zeros([len(list_of_arrays), list_of_arrays[0].ndim])
  far_corner = np.array([x.shape for x in list_of_arrays])
  leftpad  = align_pt.max(0) - align_pt
  far_corner = far_corner + leftpad
  rightpad = far_corner.max(0) - far_corner
  def f(_r):
    x,lp,rp = _r
    p = np.stack([lp,rp],axis=1)
    r = np.pad(x,pad_width=p)
    return r
  res = np.stack([f(_r) for _r in zip(list_of_arrays,leftpad,rightpad)])
  return res

def draw(tb):
  pos  = nx.multipartite_layout(tb,subset_key='time')
  cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
  # colors = np.array([tb.edges[e]['track'] for e in tb.edges])
  colors = np.array([tb.nodes[n]['track'] for n in tb.nodes])
  # return pos
  # plt.scatter(pos)
  # nx.draw(tb,pos=pos,node_size=30,edge_color=cmap(colors))
  nx.draw(tb,pos=pos,node_size=30,node_color=cmap(colors))


## Testing

def gen_ltps1():
  res = []
  N,T = 1,10
  x = np.random.rand(N,2)*10
  for i in range(T):
    x += np.random.rand(N,2)*2
    res.append(x.copy())
  return res

# def test_cele_simple_tracking(tracklets):
#   ltps = list(ndi.group_by(tracklets[:,1]).split(tracklets[:,2:]))
#   dists, parents = nn_tracking(ltps)

def check_tb_nap_lpts_consistency(tb,nap,lpts,track_id=None):
  m = nap.tracklets[:,0]==track_id
  print(nap.tracklets[m])
  print(nap.tracklets[m].shape)

  # times = nap.tracklets[m][:,1]
  # for t in times:
  #   kdt = pyKDTree(x[i]*scale)
  #   _dis, _ind = kdt.query(x[i+1]*scale, k=1, distance_upper_bound=dub)

  for n in tb.nodes:
    if tb.nodes[n]['track']==track_id:
      print(tb.nodes[n])


## Tracking Utility

def _parents2tb(parents,ltps):
  """
  parents == -1 when no parent exists
  layer == -2 when no point exists
  """
  list_of_edges = []
  for time,layer in enumerate(parents):
    if layer is []: continue
    # if np.in1d(layer,[-1]).all(): continue
    list_of_edges.extend([((time+1,n),(time,parent_id)) for n,parent_id in enumerate(layer) if parent_id!=-1])
  tb = nx.from_edgelist(list_of_edges, nx.DiGraph())
  tb = tb.reverse()

  all_nodes = [(_time,idx) for _time in np.r_[:len(ltps)] for idx in np.r_[:len(ltps[_time])]]
  tb.add_nodes_from(all_nodes)

  for v in tb.nodes: tb.nodes[v]['time'] = v[0]
  for v in tb.nodes: tb.nodes[v]['pt'] = ltps[v[0]][v[1]]
  _tb_add_track_labels(tb)
  return tb

def filter_starting_tracks_pts(tb,pts0,gtpts):
  # ipdb.set_trace()
  kdt = pyKDTree(pts0)
  _dis, _ind = kdt.query(gtpts, k=1, distance_upper_bound=None)
  roots = set((0,i) for i in _ind)
  # s0=set.union(*[c for c in nx.weakly_connected_components(tb) if len(set(c)&roots)>0])
  s0=roots
  for r in roots: s0 = s0|nx.descendants(tb,r)
  g0=tb.subgraph(s0)
  return g0

def filter_starting_tracks(tb,ltps,nap):
  m=nap.tracklets[:,1]==0
  gtpts=nap.tracklets[m][:,2:]
  g0 = filter_starting_tracks_pts(tb,ltps[0],gtpts)
  return g0

### in-place updates on tb
def _tb_add_orig_labels(tb,nap):
  origlabelmap = dict()
  tracklets    = nap.tracklets
  # group by time
  for _time,sub_tracklets in enumerate(ndi.group_by(tracklets[:,1]).split(tracklets)):
    for _idx,row in enumerate(sub_tracklets):
      origlabelmap[(_time,_idx)] = row[0] ## track id
  for n in tb.nodes:
    tb.nodes[n]['orig_trackid'] = origlabelmap[n]

### in-place updates on tb
def _tb_add_track_labels(tb):
  track_id = 1
  # source = [n for n in tb.nodes if tb.nodes[n]['time']==0]
  source = [n for n,d in tb.in_degree if d==0]
  for s in source:
    for v in nx.dfs_preorder_nodes(tb,source=s):
      tb.nodes[v]['track'] = track_id
      tb.nodes[v]['root']  = s
      if tb.out_degree[v] != 1: track_id+=1
    # tb.nodes[s]['root']  = 0

### in-place updates on tb
def relabel_tracks_from_1(tb):
  s = set(tb.nodes[n]['track'] for n in tb.nodes)
  d = {k:v+1 for v,k in enumerate(sorted(s))}
  for n in tb.nodes:
    _t = tb.nodes[n]['track']
    tb.nodes[n]['track'] = d[_t]


## Tracking from ltps (produces TrueBranching)

def nn_tracking_on_ltps(ltps=None, scale=(1,1,1), dub=None):
  """
  ltps should be time-ordered list of ndarrays w shape (N,M) with M in [2,3].
  x[t+1] matches to first nearest neib of x[t].

  points can be indexed by (time,local index ∈ [0,N_t])
  """

  dists, parents = [],[]

  for i in range(len(ltps)-1):
    Nparents = len(ltps[i])
    N = len(ltps[i+1])
    if Nparents==0:
      dists.append(np.full(N, -1))
      parents.append(np.full(N, -1))
      continue
    if N==0:
      dists.append([])
      parents.append([])
      continue

    kdt = pyKDTree(ltps[i]*scale)
    _dis, _ind = kdt.query(ltps[i+1]*scale, k=1, distance_upper_bound=dub)
    _ind = _ind.astype(np.int32)
    # if (_ind==8).sum()>0: ipdb.set_trace()
    _ind[_ind==Nparents] = -1
    dists.append(_dis)
    parents.append(_ind)

  # ipdb.set_trace()
  tb = _parents2tb(parents,ltps)

  return tb

def random_tracking_on_ltps(ltps,):
  """
  connect points at random. first try to use up all the parents, then loop back and try again.
  """
  labels  = [np.arange(len(_x)) for _x in ltps]
  parents = []
  for _time in range(len(ltps)-1):
    l = labels[_time]
    np.random.shuffle(l)
    parents.append(np.resize(l,len(labels[_time+1])))
  tb = _parents2tb(parents,ltps)
  return tb


## Type Conversions

def nap2lbep(nap):
  tracklets,graph,properties = nap.tracklets,nap.graph,nap.properties

  full_graph = graph.copy()
  for k in set(properties['root_id']): full_graph[k] = 0
  lbep = np.zeros([len(full_graph),4])

  # group by trackid
  for i,track_group in enumerate(ndi.group_by(tracklets[:,0]).split(tracklets)):
    track_id = track_group[0,0]
    lbep[i,0] = track_id
    lbep[i,1] = track_group[:,1].min() # min time for given track
    lbep[i,2] = track_group[:,1].max() # max time for given track
    x = full_graph[track_id]; #print(x)
    lbep[i,3] = x if 'int' in str(type(x)) else x[0]
  lbep = lbep.astype(np.uint)
  return lbep

def tb2nap(tb,ltps=None):
  trackid = np.array([n + (tb.nodes[n]['track'],) for n in tb.nodes])
  nodes, trackid = trackid[:,:2],trackid[:,[2]]
  idx     = np.lexsort(nodes.T[[1,0]])
  nodes,trackid = nodes[idx],trackid[idx]

  if ltps:
    _ltps = [ltps[_time][_id] for (_time,_id) in nodes]
  else:
    _ltps = [tb.nodes[(_time,_id)]['pt'] for (_time,_id) in nodes]

  tracklets = np.concatenate([trackid, nodes[:,[0]], _ltps],axis=1).astype(np.uint)
  idx = np.lexsort(tracklets[:,[1,0]].T)
  tracklets = tracklets[idx]

  graph = dict()
  for e in tb.edges:
    l0 = tb.nodes[e[0]]['track']
    l1 = tb.nodes[e[1]]['track']
    # print(l0)
    if l0!=l1: graph[l1]=l0

  root_id = []
  for n in nodes:
    r = tb.nodes[tuple(n)]['root']
    assert r!=0
    track = tb.nodes[r]['track']
    root_id.append(track)
  properties = dict(root_id=root_id)

  nap = SimpleNamespace()
  nap.tracklets = tracklets
  nap.graph = graph
  nap.properties = properties
  return nap

def nap2ltps(nap):
  ltps = list(ndi.group_by(nap.tracklets[:,1]).split(nap.tracklets[:,2:]))
  return ltps

def tb2lbep(tb):
  times = defaultdict(list)
  trackset = {tb.nodes[n]['track'] for n in tb.nodes}
  parent   = {t:0 for t in trackset}

  for n in tb.nodes:
    track = tb.nodes[n]['track']
    ps = list(tb.pred[n])
    if ps: ## if has parent
      parent_track = tb.nodes[ps[0]]['track']
      if track != parent_track:
        parent[track] = parent_track

    times[track].append(n[0])

  lbep = [[track, min(times[track]), max(times[track]), parent[track]] for track in sorted(trackset)]
  lbep = np.array(lbep)
  # idx = np.lexsort(lbep[:,[0]].T)
  # lbep = lbep[idx]
  return lbep




## TODO: do i keep this stuff?

## Reading/Writing to disk

# def _load_mantrack(path,dset,idx):
#   path = Path(path)
#   try:
#     stak = load(path/(dset+f"_GT/TRA/man_track{idx:03d}.tif"))
#   except:
#     stak = load(path/(dset+f"_GT/TRA/man_track{idx:04d}.tif"))

#   props = regionprops_table(stak, properties=('label', 'bbox', 'image', 'area'))

#   try:
#     # subtract 1 because the upper bbox is not included in the block (as is typical of python ranges)
#     props['centroid-0'] = (props['bbox-3']+props['bbox-0']-1)/2
#     props['centroid-1'] = (props['bbox-4']+props['bbox-1']-1)/2
#     props['centroid-2'] = (props['bbox-5']+props['bbox-2']-1)/2
#   except:
#     props['centroid-0'] = (props['bbox-2']+props['bbox-0']-1)/2
#     props['centroid-1'] = (props['bbox-3']+props['bbox-1']-1)/2

#   props['frame'] = np.full(props['label'].shape, idx)
#   props = pd.DataFrame(props)

#   return props

# def load_isbi2nap(path,dset,ntimes,):
#   # path = str(path) + '/'
#   path = Path(path)
#   data_df_raw = pd.concat([_load_mantrack(path,dset,idx) for idx in range(ntimes[0],ntimes[1])]).reset_index(drop=True)
#   data_df = data_df_raw.sort_values(['label', 'frame'], ignore_index=True)

#   kern = data_df.sort_values(['area'], ascending=False, ignore_index=True).loc[0,'image']
#   align_pt = np.array([x.shape for x in data_df['image']])//2
#   kern = pad_and_stack_arrays(data_df['image'], align_pt)
#   avg_kern = kern.mean(0) > 0.5

#   cols = COLUMNS
#   if '2D' in path.name: cols = cols[:-1]
#   tracklets = data_df.loc[:,cols].to_numpy().astype(np.uint)

#   lbep = np.loadtxt(path/(dset+'_GT/TRA/man_track.txt'), dtype=np.uint)
#   if lbep.ndim==1: lbep=lbep.reshape([1,4])
#   full_graph = dict(lbep[:, [0, 3]])
#   graph = {k: v for k, v in full_graph.items() if v != 0} ## weird. graph keys != set of all labels (missing root nodes). can only get root nodes from properties... i would prefer to work with tracklets + full_graph?

#   def _root(node: int):
#     """Recursive function to determine the root node of each independent component."""
#     if full_graph[node] == 0:  # we found the root
#         return node
#     return _root(full_graph[node])
#   roots = {k: _root(k) for k in full_graph.keys()}
#   properties = {'root_id': [roots[idx] for idx in tracklets[:, 0]]}

#   nap = SimpleNamespace()
#   nap.tracklets = tracklets
#   nap.graph = graph
#   nap.properties = properties
#   nap.kern = kern
#   nap.avg_kern = avg_kern
#   return nap


# from datagen import mantrack2pts
def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def save_isbi_tb(tb,info,_kern=None):

  if info.penalize_FP=='0':
    # ipdb.set_trace()
    pts_zero = np.array([tb.nodes[n]['pt'] for n in sorted(tb.nodes) if n[0]==0])
    pts_gt_zero = mantrack2pts(load(info.isbi_dir / (info.dataset+"_GT") / "TRA" / info.man_track.format(time=info.start)))
    tb = filter_starting_tracks_pts(tb,pts_zero,pts_gt_zero)
    relabel_tracks_from_1(tb)

  # ipdb.set_trace()
  nap = tb2nap(tb)
  nap.tracklets[:,1] += info.start
  if not _kern: _kern = np.ones([1,]*info.ndim)
  save_isbi(nap,_kern=_kern,shape=info.shape,savedir=info.isbi_dir/(info.dataset+"_RES"))

def eval_tb_isbi(tb,info,savedir):
  save_isbi_tb(tb,info)
  bashcmd = f"""
  localtra=/projects/project-broaddus/comparison_methods/EvaluationSoftware2/Linux/TRAMeasure
  localdet=/projects/project-broaddus/comparison_methods/EvaluationSoftware2/Linux/DETMeasure
  $localdet {info.isbi_dir} {info.dataset} {info.ndigits} {info.penalize_FP} > {savedir}/{info.dataset}_DET.txt
  cat {savedir}/{info.dataset}_DET.txt
  $localtra {info.isbi_dir} {info.dataset} {info.ndigits} > {savedir}/{info.dataset}_TRA.txt
  cat {savedir}/{info.dataset}_TRA.txt
  """
  run(bashcmd,shell=True)



def save_isbi_tb_2(tb,radius,scale,imshape,t_start,ndim,savedir,*,penalizeFP='1',mantrack_t0=None,):

  assert len(tb.nodes)>0 , "Tracks are empty"
  if penalizeFP=='0':
    # ipdb.set_trace()
    pts_zero = np.array([tb.nodes[n]['pt'] for n in sorted(tb.nodes) if n[0]==0])
    pts_gt_zero = mantrack2pts(imread(mantrack_t0))
    tb = filter_starting_tracks_pts(tb,pts_zero,pts_gt_zero)
    relabel_tracks_from_1(tb)

  # ipdb.set_trace()
  nap = tb2nap(tb)
  nap.tracklets[:,1] += t_start
  # _kern = np.ones([1,]*ndim)
  return save_isbi_2(nap, radius, scale, shape=imshape,savedir=savedir)


"""
the inverse of `isbifiles_to_napari`
sort the tracklets by time (TrackID,Time,Z,Y,X)
for each time rasterize detections using the correct labels
write the man_tracks.txt from properties alone.
"""
def save_isbi_2(nap, radius, scale, shape=None, savedir="napri2isbi_test/"):
  # t0 = pytime(); print(f"t0:{t0}")
  tracklets,graph,properties = nap.tracklets,nap.graph,nap.properties

  # ipdb.set_trace()
  savedir = Path(savedir)
  savedir.mkdir(parents=True,exist_ok=True)
  # for x in savedir.glob('mask*.tif'): x.unlink()

  lbep = nap2lbep(nap)
  np.savetxt(savedir / "res_track.txt",lbep,fmt='%d')

  ndigits = 4 if tracklets[:,1].max() > 1000 else 3
  tifname = "mask{time:03d}.tif" if ndigits==3 else "mask{time:04d}.tif"

  labelset = []
  stackset = []

  time_start = int(tracklets[:,1].min())
  time_stop  = int(tracklets[:,1].max())
  # for sub_tracklets in ndi.group_by(tracklets[:,1]).split(tracklets):


  for i in range(time_start,time_stop+1):
    # t1 = pytime(); print(f"t1:{t1}")
    m = tracklets[:,1]==i
    if m.sum()==0:
      imsave(str(savedir / tifname.format(time=i)), np.zeros(shape,dtype=np.uint16), compress=4)
      continue
    sub_tracklets = tracklets[m]
    time   = sub_tracklets[0,1]
    assert time==i
    labels = sub_tracklets[:,0]
    labelset.append(labels)
    pts    = sub_tracklets[:,2:].astype(np.int)
    # kerns  = [_kern * _id for _id in labels]
    print(f"Drawing {len(pts)} pts at t={i} . Max Label is {labels.max()}.")
    # stack = np.zeros(shape,dtype=np.uint16)
    # t2 = pytime(); print(f"t2:{t2}")
    # stack[tuple(pts.T)] = labels
    # stack  = expand_labels(stack,radius,scale)
    stack = place_label_spheres(pts,labels,shape,scale,radius)
    # t3 = pytime(); print(f"t3:{t3}")
    # stackset.append(set(np.unique(stack)))
    # t4 = pytime(); print(f"t4:{t4}")
    savename = savedir / tifname.format(time=time)

    # print(f"SIZE OF STACK {i} IS: {stack.shape}\n")
    stack = stack.astype(np.uint16)
    # t5 = pytime(); print(f"t5:{t5}")
    imsave(savename, stack, compress=4)
    # t6 = pytime(); print(f"t6:{t6}")

  return lbep, labelset, stackset





def save_isbi(nap, _kern=None, shape=(35, 512, 708), savedir="napri2isbi_test/"):
  """
  the inverse of `isbifiles_to_napari`
  sort the tracklets by time (TrackID,Time,Z,Y,X)
  for each time rasterize detections using the correct labels
  write the man_tracks.txt from properties alone.
  """
  tracklets,graph,properties = nap.tracklets,nap.graph,nap.properties

  savedir = Path(savedir)
  savedir.mkdir(parents=True,exist_ok=True)
  for x in savedir.glob('mask*.tif'): x.unlink()

  lbep = nap2lbep(nap)
  np.savetxt(savedir / "res_track.txt",lbep,fmt='%d')

  ndigits = 4 if tracklets[:,1].max() > 1000 else 3
  tifname = "mask{time:03d}.tif" if ndigits==3 else "mask{time:04d}.tif"

  labelset = []
  stackset = []

  time_start = int(tracklets[:,1].min())
  time_stop  = int(tracklets[:,1].max())
  # for sub_tracklets in ndi.group_by(tracklets[:,1]).split(tracklets):
  for i in range(time_start,time_stop+1):
    m = tracklets[:,1]==i
    if m.sum()==0:
      save(np.zeros(shape,dtype=np.uint16), savedir / tifname.format(time=i))
      continue
    sub_tracklets = tracklets[m]
    time   = sub_tracklets[0,1]
    assert time==i
    labels = sub_tracklets[:,0]
    labelset.append(labels)
    pts    = sub_tracklets[:,2:].astype(np.int)
    kerns  = [_kern * _id for _id in labels]
    stack  = conv_at_pts_multikern(pts,kerns,shape).astype(np.uint16)
    stack  = expand_labels(stack,25)
    stackset.append(set(np.unique(stack)))
    savename = savedir / tifname.format(time=time)
    # ipdb.set_trace()
    save(stack, savename)

  return lbep, labelset, stackset

def save_permute_existing(tb, info, savedir):
  savedir = Path(savedir)

  for _time in np.r_[info.start:info.stop]:
    name = info.isbi_dir/(info.dataset+f"_GT/TRA/")/info.man_track.format(time=_time)
    # WARNING: _time is pseudotime, which doesn't correspond with actual timestring for PSC PhC-C2DL-PSC datasets!!
    lab = load(name)
    mapping = {tb.nodes[n]['orig_trackid']:tb.nodes[n]['track'] for n in tb.nodes if n[0]==_time}
    lab2 = relabel_from_mapping(lab,mapping).astype(np.uint16)
    save(lab2, savedir / info.maskname.format(time=_time))

  # lbep = nap2lbep(tb2nap(tb,nap2ltps(nap)))
  lbep = tb2lbep(tb)
  lbep[:,[1,2]] = lbep[:,[1,2]]+info.start
  np.savetxt(savedir / "res_track.txt",lbep,fmt='%d')
  return lbep

def compare_all_labelsets(nap=None,lbep=None,tradir=None,tb=None):

  all_lsds = dict()

  resdir = Path(resdir)
  if tradir:
    lbep2 = np.loadtxt(resdir / 'res_track.txt').astype(np.uint)

  lbep_lsd = lbep2lsd(lbep)
  tradir_lsd = TRAdir2lsd(resdir)

def compare_lsds(lsd1,lsd2):
  for k in sorted(set.union(set(lsd1.keys()),set(lsd2.keys()))):
    print(k, len(lsd1[k]-lsd2[k]), len(lsd2[k]-lsd1[k]))

def nap2lsd(nap):
  tracklets = nap.tracklets
  lsd = dict()
  for x in ndi.group_by(tracklets[:,1]).split(tracklets):
    lsd[x[0,1]] = set(x[:,0])
  return lsd

def TRAdir2lsd(tradir,):
  tradir = Path(tradir)
  lsd = dict()
  for name in sorted(tradir.glob("*.tif")):
    time = int(re.search(r"(\d{3,4})\.tif", str(name)).group(1))
    img  = load(name)
    lsd[time] = set(np.unique(img))
  return lsd

def load_lbep(name):
  lbep = np.loadtxt(name).astype(np.uint)
  if lbep.ndim == 1: lbep = lbep.reshape([1,4])
  return lbep

def lbep2lsd(lbep):
  """
  lbep -> label set dict
  """
  from collections import defaultdict
  mintime, maxtime = lbep[:,1].min(), lbep[:,2].max()
  lsd = defaultdict(set)
  # for time in np.r_[mintime:maxtime+1]:
  #   lsd[time] = {x[0] for x in lbep if x[1] <= time <= x[1]}
  for row in lbep:
    for time in np.r_[row[1]:row[2]+1]:
      lsd[time].add(row[0])

  return lsd

def load_and_compare_lsds(Fluodir,ds='01'):
  Fluodir  = Path(Fluodir)
  d = SimpleNamespace()
  # d.lsd_orig_1 = lbep2lsd(load_lbep(Fluodir / f'{ds}_GT/TRA/man_track.txt'))
  # d.lsd_orig_2 = TRAdir2lsd(Fluodir / f'{ds}_GT/TRA/')
  d.lbep  = lbep2lsd(load_lbep(Fluodir / f'{ds}_RES/res_track.txt'))
  d.labs  = TRAdir2lsd(Fluodir / f'{ds}_RES/')
  # from itertools import combinations
  # for x,y in combinations([d.lsd_orig_1,d.lsd_orig_2,d.lsd_new_1,d.lsd_new_2,],2):
  #   compare_lsds(x,y)
  return d


# def lsd2lbep(lsd):
#   res = defaultdict(list)
#   for time,lset in lsd.items():
#     for l in lset:
#       res[l].append(time)


# if __name__=='__main__':
#   x = gen_ltps1()
#   x[3] = np.array([]).reshape([0,2])
#   print(x)
#   tb = nn_tracking_on_ltps(x,scale=(1,1))
#   print(tb.edges)




"""bash
isbidir=/Users/broaddus/Desktop/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/
localtra=/Users/broaddus/Downloads/EvaluationSoftware_1/Mac/TRAMeasure
rm "$isbidir"01_RES/*
cp -r napri2isbi_test/* "$isbidir"01_RES/
time $localtra $isbidir 01 3

isbidir=/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/
localtra=/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/TRAMeasure
cp -r nap2isbi/ "$isbidir"01_RES/
time $localtra $isbidir 01 3
"""


"""
TODO:
- make it work for 2D/3D better
- replace pandas with numpy_indexed?
-

ways to specify nodes:
- tuple of (time, local label in lab-img) [potentially local label correspond to tracks across time]
- global index into row of tracklets
-



"""
