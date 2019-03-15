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
                allow_div        = False,
                allow_exit       = False,
                allow_appear     = False,
                do_velcorr       = True,
                neib_edge_cutoff = 40,
                velgrad_scale    = 20,
                on_edges         = None,
                ):

        graph_cost_stats = []
        self.knn_n              = knn_n
        self.knn_dub            = knn_dub
        self.neib_edge_cutoff   = neib_edge_cutoff
        self.graph_cost_stats   = graph_cost_stats
        self.do_velcorr         = do_velcorr
        self.velgrad_scale      = velgrad_scale
        self.on_edges           = on_edges
        self.edgecost           = edgecost
        self.vertcost           = vertcost
        self.allow_div          = allow_div
        self.allow_exit         = allow_exit
        self.allow_appear       = allow_appear

        # self.on_edges = [((1,93), (2,100)),
        #             ((3,119), (4,96)),
        #             ((3,134), (4,107))]

    def nhls2graph(self, nhls):
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

    def graph2pulp(self, graph, nhls):
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
        if self.do_velcorr:
            vcterm = self.add_velocity_correlation(prob, graph, nhls, edgevars)
            prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm) + pulp.lpSum(vcterm), "Objective"
        else:
            prob += pulp.lpSum(edgeterm) + pulp.lpSum(vertterm), "Objective"

        prob.writeLP('tracker.lp')
        prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 200), ('OptimalityTol', 1e-7)]))
        # prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit', 100), ('ResultFile','coins.sol'), ('OptimalityTol', 1e-2)]))
        # prob.solve(pulp.PULP_CBC_CMD(options=['-sec 300']))
        return prob, vertvars, edgevars
  
    def nhls2tracking(self, nhls):
        graph = self.nhls2graph(nhls)
        nucdict = nhl_tools.nhls2nucdict(nhls)

        prob, vv, ev = self.graph2pulp(graph, nhls)

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

    def build_neighbor_graphs(self, nhls):
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

    def add_velocity_correlation(self, prob, graph, nhls, edgevars):
        ## velocity correlation variables
        neighbor_graphs = self.build_neighbor_graphs(nhls)
        nuc_dict = nhl_tools.nhls2nucdict(nhls)
        neighbor_vars = []
        for ng in neighbor_graphs[:-1]:
            for e in ng.edges:
                u,v = e
                for ui in graph[u]:
                    for vi in graph[v]:
                        if ui!=vi:
                            neighbor_vars.append((u,ui,v,vi))
        neighbor_vars_pulp = pulp.LpVariable.dicts('velocitygrad', neighbor_vars, lowBound=0, upBound=1, cat=pulp.LpBinary)
        costs = self.velocity_grad_cost_list(nuc_dict, neighbor_vars)
        x = np.array(costs)
        self.graph_cost_stats.append((x.mean(), x.min(), x.max(), x.std()))
        neighbor_vars_term = [costs[i]*neighbor_vars_pulp[vp] for i,vp in enumerate(neighbor_vars)]

        ## velocity correlation constraints
        for vp in neighbor_vars:
            (u,ui,v,vi) = vp
            prob += edgevars[(u,ui)] + edgevars[(v,vi)] >= 2 * neighbor_vars_pulp[vp], ''
            prob += edgevars[(u,ui)] + edgevars[(v,vi)] <= 1 + neighbor_vars_pulp[vp], ''

        return neighbor_vars_term

    def velocity_grad_cost_list(self, nucdict, neighbor_vars):
        xu  = np.array([nucdict[vp[0]]['centroid'] for vp in neighbor_vars])
        xui = np.array([nucdict[vp[1]]['centroid'] for vp in neighbor_vars])
        xv  = np.array([nucdict[vp[2]]['centroid'] for vp in neighbor_vars])
        xvi = np.array([nucdict[vp[3]]['centroid'] for vp in neighbor_vars])
        dxu = xui - xu
        # dxu = dxu / np.linalg.norm(dxu, axis=1, keepdims=True)
        dxv = xvi - xv
        # dxv = dxv / np.linalg.norm(dxv, axis=1, keepdims=True)
        res = np.linalg.norm(dxu - dxv, axis=1)/self.velgrad_scale - 1.0
        # res = -1.0*(dxu*dxv).sum(1)
        # res[:] = 0
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
    graph = trackfactory.nhls2graph(nhls)
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

def lineagelabelmap2(tb,tv):
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

