import numpy as np
from stackview.stackview import Stack
import matplotlib

def draw_flow_current(iss, flowlist):
    draw_flow(iss, flowlist[:,min(iss.stack.shape[0]-2, iss.idx[0])])

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
        ax.arrow(x0,y0,vy,vx,width=0.1,head_width=1, head_length=1.0, fc='w', ec='w')
        # ax.plot([x0,x0+vx],[y0, y0+vy], 'k') #  head_width=4, fc='w', ec='w')

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

def draw_arrows_current_3d(iss, tr, dz=10):
    t,z = iss.idx
    arrows = tr.al[min(iss.stack.shape[0]-2, t)]
    arrows = [a for a in arrows if z-dz < a[0][0] < z+dz]
    draw_arrows(iss, arrows, lambda x: x[[1,2]])

def load_tracks_into_spimagine(w, img, tr):
    "n is nuclear hypothesis in (time, id) format."
    wcc = sorted(nx.weakly_connected_components(tr.tb),key=lambda x:len(x))
    nucdict = nhl_tools.nhls2nucdict(nhls)
    img2 = np.pad(img, (0,1,40,40,0), mode='constant')
    best = wcc[-2]
    slices = [nhl_tools.nuc2slices_centroid(nucdict[n], (1,40,40), (1,40,40)) for n in best]
    patches = np.array([img2[2:][i][slices[i]] for i in range(len(slices))])

def plot_projected_trajectories():
    for bes in wcc[-3:]:
        trajectory = np.array([nucdict[n]['centroid'] for n in bes])
        plt.plot(trajectory[:,2], trajectory[:,1])

def draw_circles(iss, nhls, dz=10):
    t,z = iss.idx
    ax = iss.fig.gca()
    ax.patches = []
    nhl = nhls[t]
    nhl = [n for n in nhl if z-dz < n['centroid'][0] < z+dz]
    for nuc in nhl:
        x,y = nuc['centroid'][2], nuc['centroid'][1]
        p = matplotlib.patches.Circle((x,y),radius=20,fill=False)
        ax.add_patch(p)

def draw_both(iss, nhls, tr, dz=10):
    draw_circles(iss, nhls, dz)
    draw_arrows_current_3d(iss, tr, dz)


    # offsets = np.array([n['centroid'] for n in nhl])[:,[2,1]]
    # coll = matplotlib.collections.CircleCollection([400]*len(nhl), transOffset=offsets)



