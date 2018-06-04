import numpy as np
from stackview.stackview import Stack

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

def draw_arrows_current_3d(iss, tr):
    t,z = iss.idx
    arrows = tr.al[min(iss.stack.shape[0]-2, t)]
    arrows = [a for a in arrows if z-10 < a[0][0] < z+10]
    draw_arrows(iss, arrows, lambda x: x[[1,2]])