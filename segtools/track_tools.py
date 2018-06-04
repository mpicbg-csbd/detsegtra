from tabulate import tabulate
import numpy as np
import networkx as nx
import pulp
from collections import defaultdict, Counter
from sortedcollections import SortedDict
from PIL import Image, ImageDraw
from multipledispatch import dispatch
import matplotlib.pyplot as plt
import scipy.spatial as sp
from numba import jit

from . import scores_dense as ss
from . import nhl_tools
from . import graphmatch as gm
from . import color
from .python_utils import print_sorted_counter

from stackview.stackview import Stack

## A single function to do everything

def build_neighbor_graphs(nhls):
    nucdict = nhls2nucdict(nhls)
    points = lambda i: np.array([n['centroid'] for n in nhls[i]])
    labels = lambda i: [(i, n['label']) for n in nhls[i]]
    def f(points, labels, **kwargs):
        vor = sp.Voronoi(points)
        g = nx.from_edgelist([(labels[x], labels[y]) for x,y in vor.ridge_dict.keys()])
        remove_long_edges(g, nucdict, **kwargs)
        return g
    neighbor_graphs = [f(points(i), labels(i), cutoff=150) for i in range(len(nhls))]
    return neighbor_graphs

def nhls2tracking(nhls, **kwargs):
    graph = nhls2graph(nhls)
    nucdict = nhls2nucdict(nhls)

    # on_edges = [((1,93), (2,100)),
    #             ((3,119), (4,96)),
    #             ((3,134), (4,107))]

    prob, vv, ev = graph2pulp(graph, nhls, **kwargs)

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

## utilities and data munging


def filter_nhls(nhls):
    def fil(n):
        if 3 < np.log2(n['area']):
            return True
        return False
    nhls2 = []
    for i,nhl in enumerate(nhls):
        nhl = [n for n in nhl if fil(n)]
        nhls2.append(nhl)
    return nhls2

def nhl2matrix(nhl):
    if False:
        flat = [[n['area'],
                n['bbox'][0],
                n['bbox'][1],
                n['bbox'][2],
                n['bbox'][3],
                n['centroid'][0],
                n['centroid'][1]] for n in nhl]
    flat = [[n['centroid'][0],
             n['centroid'][1]] for n in nhl]
    labels = [n['label'] for n in nhl]
    return np.array(flat), np.array(labels)

def nhls2mats(nhls):
    return [nhl2matrix(nhl)[0] for nhl in nhls]

def nhls2mats_labs(nhls):
    return zip(*[nhl2matrix(nhl) for nhl in nhls])

def mats2bips(mats):
    bips = [gm.centroid_bipartite(mats[i], mats[i+1], dub=50) for i in range(len(mats)-1)]
    return bips

def mats_labs2bips(mats, labs):
    bips = [gm.connect_points_digraph(mats[i], mats[i+1], lx=i, ly=i+1, labels_x=labs[i], labels_y=labs[i+1], k=3, dub=100) for i in range(len(mats)-1)]
    return bips

def nhls2bips(nhls):
    "assumes consecutive time points starting from 0."
    mats, labs = nhls2mats_labs(nhls)
    bips = mats_labs2bips(mats, labs)
    return bips

def bips2graph(bips):
    graph = bips[0]
    for bip in bips[1:]:
        graph = nx.compose(graph, bip)
    return graph

def nhls2graph(nhls):
    bips = nhls2bips(nhls)
    graph = bips2graph(bips)
    return graph

def tb_from_labs(labs):
    "given a labeled timeseries with lineage consistent cell-id's build the associated DiGraph + hyp image"
    # for i,lab in enumerate(labs):
    pass



def labs_imgs2graph(labs, imgs):
    nhls = labs2nhls(labs, imgs)
    graph = nhls2graph(nhls)
    return graph

def nhls2nhldicts(nhls):
    def f(lis):
        d = dict()
        for n in lis:
            d[n['label']] = n
        return d
    return [f(nhl) for nhl in nhls]

def nhls2nucdict(nhls, f=lambda x: x):
    d = dict()
    for i, nhl in enumerate(nhls):
        for n in nhl:
            d[(i, n['label'])] = f(n)
    return d

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

## graph manipulation

def remove_long_edges(g, nucdict, cutoff=200):
    badedges = []
    for e in g.edges:
        v,u = e
        x1,x2 = np.array(nucdict[v]['centroid']), np.array(nucdict[u]['centroid'])
        dr = np.linalg.norm(x1-x2)
        if dr > cutoff:
            badedges.append(e)
    g.remove_edges_from(badedges)

## PuLP stuff. Solve the problem.

favstats = []

def graph2pulp(graph, nhls, velcorr=True, on_edges=None):
    prob = pulp.LpProblem("Assignment Problem", pulp.LpMinimize)

    nuc_dict = nhls2nucdict(nhls)

    ## vertex and edge variables
    vertvars = pulp.LpVariable.dicts('verts', graph.nodes, lowBound=0, upBound=1, cat=pulp.LpBinary)
    vertcostdict = {n:vertcost(nuc_dict, n) for n in graph.nodes}
    vertterm = [vertcostdict[n]*vertvars[n] for n in graph.nodes]

    edgevars = pulp.LpVariable.dicts('edges', graph.edges, lowBound=0, upBound=1, cat=pulp.LpBinary)
    edgecostdict = {(n1,n2):edgecost(nuc_dict, n1, n2) for (n1,n2) in graph.edges}
    edgeterm = [edgecostdict[e]*edgevars[e] for e in graph.edges]

    print("VERT & EDGE COSTS")
    x = np.array(list(vertcostdict.values()))
    print(x.mean(), x.min(), x.max(), x.std())
    x = np.array(list(edgecostdict.values()))
    print(x.mean(), x.min(), x.max(), x.std())


    ## vertex and edge constraints
    for n in graph.nodes:
        e_out = [edgevars[(n,v)] for v in graph[n]]
        if len(e_out) > 0:
            prob += pulp.lpSum(e_out) <= 1 * vertvars[n], ''
    for n in graph.nodes:
        e_in = [edgevars[(v,n)] for v in graph.pred[n]]
        if len(e_in) > 0:
            prob += pulp.lpSum(e_in) <= vertvars[n], ''

    if on_edges:
        for e in on_edges:
            prob += edgevars[e] == 1, ''

    ## objective
    if velcorr:
        vcterm = add_velocity_correlation(prob, graph, nhls, edgevars)
        prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm) + pulp.lpSum(vcterm), "Objective"
    else:
        prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm), "Objective"

    prob.writeLP('tracker.lp')
    prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 200), ('OptimalityTol', 1e-7)]))
    # prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 100), ('ResultFile','coins.sol'), ('OptimalityTol', 1e-2)]))
    # prob.solve(pulp.PULP_CBC_CMD(options=['-sec 300']))
    return prob, vertvars, edgevars

def add_velocity_correlation(prob, graph, nhls, edgevars):
    ## velocity correlation variables
    neighbor_graphs = build_neighbor_graphs(nhls)
    nuc_dict = nhls2nucdict(nhls)
    neighbor_vars = []
    for ng in neighbor_graphs[:-1]:
        for e in ng.edges:
            u,v = e
            for ui in graph[u]:
                for vi in graph[v]:
                    if ui!=vi:
                        neighbor_vars.append((u,ui,v,vi))
    neighbor_vars_pulp = pulp.LpVariable.dicts('velocitygrad', neighbor_vars, lowBound=0, upBound=1, cat=pulp.LpBinary)
    costs = velocity_grad_cost_list(nuc_dict, neighbor_vars)
    print("Cost List\n")
    x = np.array(costs)
    print(x.mean(), x.min(), x.max(), x.std())
    neighbor_vars_term = [costs[i]*neighbor_vars_pulp[vp] for i,vp in enumerate(neighbor_vars)]

    ## velocity correlation constraints
    for vp in neighbor_vars:
        (u,ui,v,vi) = vp
        prob += edgevars[(u,ui)] + edgevars[(v,vi)] >= 2 * neighbor_vars_pulp[vp], ''
        prob += edgevars[(u,ui)] + edgevars[(v,vi)] <= 1 + neighbor_vars_pulp[vp], ''

    return neighbor_vars_term

def velocity_grad_cost_list(nucdict, neighbor_vars):
    xu  = np.array([nucdict[vp[0]]['centroid'] for vp in neighbor_vars])
    xui = np.array([nucdict[vp[1]]['centroid'] for vp in neighbor_vars])
    xv  = np.array([nucdict[vp[2]]['centroid'] for vp in neighbor_vars])
    xvi = np.array([nucdict[vp[3]]['centroid'] for vp in neighbor_vars])
    dxu = xui - xu
    # dxu = dxu / np.linalg.norm(dxu, axis=1, keepdims=True)
    dxv = xvi - xv
    # dxv = dxv / np.linalg.norm(dxv, axis=1, keepdims=True)
    res = np.linalg.norm(dxu - dxv, axis=1)/20.0 - 1.0
    # res = -1.0*(dxu*dxv).sum(1)
    # res[:] = 0
    return res

def edgecost(nucdict, n1, n2):
    v1 = nucdict[n1]
    v2 = nucdict[n2]
    da = abs(v1['area'] - v2['area']) / min(v1['area'], v2['area']) - 0.1
    
    dx = np.array(v1['centroid']) - np.array(v2['centroid'])
    dxnorm = dx / 20
    # distcost = (dxnorm*dxnorm).sum() - 1.0
    distcost = np.linalg.norm(dxnorm) - 1.0
    # return 0.0
    return distcost

def vertcost(nucdict, n):
    return -1

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

def stats_tr(tr):
    stats(tr.tv, tr.te, tr.tb)

def arrowlist(tb, tv, nuc_dict):
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


## coloring and drawing

def recolor_every_frame(lab, cm):
    # labr = np.zeros(lab.shape + (3,))
    labr = []
    for i in range(lab.shape[0]):
        labr.append(color.apply_mapping(lab[i], cm[i]))
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

def draw_flow_current(iss, flowlist):
    draw_flow(iss, flowlist[min(iss.stack.shape[0]-2, iss.idx[0])])

def draw_flow(iss, flow):
    a,b,c = flow.shape
    assert a==2
    ax = iss.fig.gca()
    ax.artists = []
    if iss.stack.ndim == 3:
        wz,wy,wx = iss.stack.shape
    elif iss.stack.ndim == 4:
        wz,wy,wx,_ = iss.stack.shape
    dy,dx = wy//b, wx//c
    flow = flow.reshape((2,-1)).T
    origins = np.indices((b,c)).reshape((2,-1)).T
    for i in range(b*c):
        y0,x0 = origins[i]
        y0,x0 = y0*dy + dy//2, x0*dx + dx//2
        vy,vx = flow[i]
        ax.arrow(x0,y0,vy,vx,head_width=4, fc='w', ec='w')

def draw_arrows_current(iss, tr):
    draw_arrows(iss, tr.al[min(iss.stack.shape[0]-2, iss.idx[0])])

def draw_arrows(iss_or_ax, arrows, trans=lambda x: x[[0,1]]):
    if type(iss_or_ax) is Stack:
        ax = iss_or_ax.fig.gca()
    else:
        ax = iss_or_ax
    ax.artists = []
    for i in range(len(arrows)):
        y0,x0 = trans(arrows[i][0])
        y1,x1 = trans(arrows[i][1])
        ax.arrow(x0,y0,x1-x0,y1-y0,head_width=1, fc='w', ec='w')

def clear_arrows(iss):
    ax = iss.fig.gca()
    ax.artists = []

def remove_top_image_from_iss(iss):
    del iss.fig.axes[0].images[-1]

## Analysis: histograms, stats, etc

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

## Utils

def minmax01norm(img):
    return (img-img.min())/(img.max()-img.min())
    # return img

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

# def pipeline(*steps):
#     return reduce(lambda f,g: g(f), steps)

def set_track_time(tr, time):
    tb2.add_nodes_from([(v[0]+time,v[1]) for v in tr.tb.nodes])
    tv2 = [(v[0]+time,v[1]) for verts in tr.tv for v in verts]
    te2 = [((v[0]+time,v[1]), (u[0]+time, u[1])) for u,v in tr.te]
    tb2 = nx.from_edgelist([((v[0]+time,v[1]), (u[0]+time, u[1])) for u,v in tr.tb.edges], create_using=nx.DiGraph())
    return tb2

def compose_trackings(tracklist, nhls):
    tvs = [[(0, v[1]) for v in tracklist[0].tv[0]]]
    for i,tr in enumerate(tracklist):
        tvs.append([(i+1, v[1]) for v in tr.tv[1]])
    tes = [[((i,u[1]), (i+1,v[1])) for u,v in tr.te[0]] for i,tr in enumerate(tracklist)]
    tb = true_branching(tvs,tes)
    cm = lineagecolormaps(tb, tvs)
    # nhls = filter_nhls(nhls)
    nucdict = nhls2nucdict(nhls)
    graph = nhls2graph(nhls)
    al = arrowlist(tb, tvs, nucdict)
    tr = Tracking(graph, tb, tvs, tes, al, cm)
    return tr

def arrows_current_3d(iss, tr):
    t,z = iss.idx
    arrows = tr.al[min(iss.stack.shape[0]-2, t)]
    arrows = [a for a in arrows if z-10 < a[0][0] < z+10]
    draw_arrows(iss, arrows, lambda x: x[[1,2]])