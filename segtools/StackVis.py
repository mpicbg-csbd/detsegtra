import numpy as np
from vispy import app, gloo, visuals, scene
from vispy.visuals.transforms import STTransform, NullTransform

class StackVis1(app.Canvas):
    def __init__(self, img):
        app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        # canvas = scene.SceneCanvas(keys='interactive')
        # canvas.show()

        # self.stack = img
        # self.colorchan = True if img.shape[-1] in {1,2,3} else False
        # self.tup = np.asarray((0,)*(img.ndim-3) if self.colorchan else (0,)*(img.ndim-2))
        # self.activedim = 0

        # rgb = np.random.randint(0,255,(400,400,3)).astype(np.uint8)
        rgb = img[2]
        self.image = visuals.ImageVisual(rgb, interpolation='nearest', method='subdivide')
        
        # view = self.central_widget.add_view()
        # self.image.parent = view.
        # view.camera = scene.PanZoomCamera(aspect=1)
        # view.camera.set_range()
        
        # self.view = view
        # self.image = image

        self.show()

    def on_draw(self, event):
        gloo.clear('black')
        self.image.draw()
        # x = self.stack[tuple(self.tup)]
        # if   self.colorchan and x.shape[-1]==2: x = x[...,[0,1,1]]*[1,0.5,0.5]
        # elif self.colorchan and x.shape[-1]==1: x = x[...,[0,0,0]]
        # self.image.set_data(x)
        # self.image.draw() 

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        # self.line.transforms.configure(canvas=self, viewport=vp)
        self.image.transforms.configure(canvas=self, viewport=vp)

    def on_key_press(self, event):
        pass
        # if event.text == 'j':
        #   self.tup[self.activedim] = (self.tup[self.activedim] - 1) % self.stack.shape[self.activedim]
        # if event.text == 'i':
        #   self.tup[self.activedim] = (self.tup[self.activedim] + 1) % self.stack.shape[self.activedim]
        # self.update()

class StackVis2(scene.canvas.SceneCanvas):
    def __init__(self, img):
        # scene.canvas.SceneCanvas.__init__(self, keys='interactive', size=(800, 800))
        super(scene.canvas.SceneCanvas,self).__init__(self, keys='interactive', size=(800, 800))
        # canvas = scene.SceneCanvas(keys='interactive')
        # canvas.show()

        # self.stack = img
        # self.colorchan = True if img.shape[-1] in {1,2,3} else False
        # self.tup = np.asarray((0,)*(img.ndim-3) if self.colorchan else (0,)*(img.ndim-2))
        # self.activedim = 0
        self.unfreeze()

        rgb = img[2]
        rgb = np.random.randint(0,255,(400,400,3)).astype(np.uint8)
        view = self.central_widget.add_view()
        view.camera = scene.PanZoomCamera(aspect=1)
        view.camera.set_range()
        self.image = scene.visuals.Image(rgb, interpolation='nearest', parent=view.scene, method='subdivide')
        
        self.view = view
        # self.image = image

        self.show()

    def on_draw(self, event):
        gloo.clear('black')
        # x = self.stack[tuple(self.tup)]
        # if   self.colorchan and x.shape[-1]==2: x = x[...,[0,1,1]]*[1,0.5,0.5]
        # elif self.colorchan and x.shape[-1]==1: x = x[...,[0,0,0]]
        # self.image.set_data(x)
        # self.image.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        # self.line.transforms.configure(canvas=self, viewport=vp)
        # self.image.transforms.configure(canvas=self, viewport=vp)

    def on_key_press(self, event):
        # if event.text == 'j':
        #   self.tup[self.activedim] = (self.tup[self.activedim] - 1) % self.stack.shape[self.activedim]
        # if event.text == 'i':
        #   self.tup[self.activedim] = (self.tup[self.activedim] + 1) % self.stack.shape[self.activedim]
        self.update()

class StackVis(object):
    """
    iss = StackVis(ndarray)
    iss.stack is the original ndarray
    iss.image is the vispy image
    iss.image.clim are image bounds. in tuple(float,float) or 'auto'
    iss.image.cmap is the colormap. most matplotlib colormaps apply?
    """
    def __init__(self, img):

        canvas = scene.SceneCanvas(keys='interactive')
        self.canvas = canvas

        self.colorchan = True if img.shape[-1] in {1,2,3} else False
        self.tup = np.asarray((0,)*(img.ndim-3) if self.colorchan else (0,)*(img.ndim-2))
        self.activedim = 0
        self.stack = img
        self.autoclim = True

        # Set up a viewbox to display the image with interactive pan/zoom
        view = canvas.central_widget.add_view()
        view.camera = scene.PanZoomCamera(aspect=1)
        view.camera.flip = (0,1,0) ## necessary for image coords and scene coords to align
        self.view = view

        x = self.state2img()
        self.x = x
        print(x.shape)
        print(x.min(), x.max())
        # self.shape = x.shape
        image = scene.visuals.Image(x, interpolation='nearest', parent=view.scene, method='subdivide',clim='auto')
        self.image = image
        self.tf = image.get_transform(map_from='canvas', map_to='visual')

        canvas.show()
        self.reset_view()

        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.text == 'j':
                self.tup[self.activedim] = (self.tup[self.activedim] - 1) % self.stack.shape[self.activedim]
            elif event.text == 'i':
                self.tup[self.activedim] = (self.tup[self.activedim] + 1) % self.stack.shape[self.activedim]
            elif event.text in {'1','2','3'}:
                self.activedim = int(event.text)-1
                print('active dim: ', self.activedim)
            print('\r Slice: '+str(self.tup)+' '*5, end="")
            self.update()

        @canvas.events.mouse_press.connect
        def on_mouse_press(event):
            x,y,_,_ = self.tf.map(event.pos)
            x,y = int(x),int(y)
            img = self.x
            print(event.pos, x, y)
            print(img[min(y,img.shape[0]-1), min(x,img.shape[1]-1)])

    def reset_view(self):
        x = self.x
        self.canvas.size = x.shape[1], x.shape[0]
        self.view.camera.set_range(margin=0)

    def state2img(self):
        x = self.stack[tuple(self.tup)].copy()
        if   self.colorchan and x.shape[-1]==2: x = x[...,[0,1,1]]*[1,0.5,0.5]
        elif self.colorchan and x.shape[-1]==1: x = x[...,[0,0,0]]
        # print(x.dtype, x.max())
        return x

    def update(self):
        x = self.state2img()
        self.x = x
        if self.autoclim: self.image.clim = x.min(),x.max()
        self.image.set_data(x)
        self.canvas.update()


def build_arrows(heads_tails,zpos=-1,color='white',**kwargs):
    """
    shape = n_arrows, head|tail, n_dim
    n_dim is 2 or 3
    """
    assert heads_tails.ndim == 3

    defaults = dict(
        arrow_size=7,
        arrow_type='curved',
        antialias =False,
        width=3,
        )
    defaults = {**defaults, **kwargs}

    ht = heads_tails
    n = heads_tails.shape[0]
    d = heads_tails.shape[-1]
    if d==2:
        ht = ht[:,:,[0,1,1]]
        ht[:,:,-1] = zpos
        d+=1

    arrows = scene.visuals.Arrow(
                pos=ht.reshape((n*2,d)),
                color=color,
                connect='segments',
                arrows=ht.reshape((n,2*d)),
                arrow_color=color,
                **defaults,
            )
    return arrows
