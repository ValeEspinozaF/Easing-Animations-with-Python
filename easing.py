import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, rc, quiver
rc('animation', html='html5')
from IPython.display import HTML, Image
from itertools import groupby
from matplotlib.patches import Polygon



class Eased:

    def __init__(self, data, in_t=[], wrap=True):
        """
        Create a new instance of the class Eased().

        Parameters
        ----------
        data : numpy.array or pandas.DataFrame
            Input data to create a new instance of the Eased class. The data must 
            have as many rows as keyframes and twice as many columns as the object 
            coordinates, i.e. the columns are consecutive pairs of x- and y-coordinates.
        in_t : list, optional
            List of integers, where each integer is related to the rows of the input data. 
            The easing will be applied to the rows specified in the `in_t` parameter, 
            in the order given. If `in_t` is not specified ([], empty list), the easing 
            will be applied to all rows in order. The default is [].
        wrap : boolean, optional
            Determines whether to wrap the animation back to the initial keyframe. 
            The default is True.
            
        Returns
        -------
        None.

        """
        
        if not isinstance(in_t, list):
            raise TypeError("in_t must be a list")
            
        if not all(isinstance(item, int) for item in in_t):
            raise TypeError("in_t must be a list of integers")
        
        if isinstance(data, pd.DataFrame):
            
            if data.dtypes.nunique() != 1:
                raise TypeError("Data must have the same data type")
            
            self.columns = data.columns
            
            if not in_t:
                in_t = np.arange(len(data.index.values)) 
            
            labels = data.index.values[in_t]
            
            """
            if isinstance(data.loc[data.index[0]][0], plt.Rectangle):   # Get this documented
                
                coords = np.zeros((len(data), 4*2))
                for z_, row in data.iterrows():
                    z = list(data.index).index(z_)
                    x0, y0 = row[0].get_xy()
                    w = row[0].get_width()
                    h = row[0].get_height()
                    coords[z, 0*2], coords[z, 0*2+1] = x0, y0
                    coords[z, 1*2], coords[z, 1*2+1] = x0, y0+h
                    coords[z, 2*2], coords[z, 2*2+1] = x0+w, y0+h
                    coords[z, 3*2], coords[z, 3*2+1] = x0+w, y0
                
                data = coords
                
                
            elif isinstance(data.loc[data.index[0]][0], quiver.Quiver): # Get this documented
                
                coords = []
                for z_, row in data.iterrows():
                    coords_z = []
                    z = list(data.index).index(z_)
                    xy0 = row[0].XY
                    w0 = row[0].U
                    h0 = row[0].V
                    for i, xy, w, h in zip(range(len(xy0)), xy0, w0, h0):
                        coords_z.extend([xy[0], xy[1], w, h])
                    
                    coords.append(coords_z)
                data = np.array(coords)
            """    
            
            if np.issubdtype(data.dtypes[0], np.integer):
                data = data.astype('float')
                data = data.values
                
            elif np.issubdtype(data.dtypes[0], np.floating):
                data = data.values            
            
        
        elif isinstance(data, np.ndarray):
            
            if not in_t:
                in_t = np.arange(np.shape(data)[0])
                
            labels = [""] * len(in_t) # with new approach, otherwise +1
            
        

        else:
            print('\033[91m' + "Data is unrecognized type : must be either a numpy Array or pandas DataFrame")
        
        
        if wrap:
            in_t = np.append(in_t, in_t[:1])
            labels = np.append(labels, labels[:1])
            
            
        self.labels = labels
        self.n_dims = len(np.shape(data))
        self.int_t = in_t
        self.data = data
        
    
    def _assign_main_frames(self):
        states = np.unique(self.int_t)
        self.main_frames = []
        for state in states:
            idx = np.where(self.int_t == state)[0][0]
            frame = self.n_steps*idx - 1
            if frame == -1: frame = 0
            self.main_frames.append((frame, state))
    
    
    def _save_main_frames(self, save_main_frames, destination, z):
        idx = list(zip(*self.main_frames))[0].index(z)
        suffix = str(list(zip(*self.main_frames))[1][idx])
        if isinstance(save_main_frames, str):
            plt.savefig(save_main_frames + suffix + ".png")
        elif destination:
            path = '.'.join(destination.split('.')[:-1])
            plt.savefig(path + suffix + ".png")
        else:
            plt.savefig(suffix + ".png")
    
    
    def _set_label(self, i, z, start, end, delta, fpt):
        # Set state labels based on coordinate position
        if self.int_t[i] != self.int_t[i+1] and delta != 0.0: 
            for frame in range(self.n_steps):
                
                if abs(self.eased[frame + fpt*i, z] - start) < abs(self.eased[frame + fpt*i, z] - end):
                    self.frame_label[frame + fpt*i] = self.labels[i]
                else:
                    self.frame_label[frame + fpt*i] = self.labels[i + 1]
                
        elif self.int_t[i] == self.int_t[i+1]:
            for frame in range(self.n_steps):
                self.frame_label[frame + fpt*i] = self.labels[i+1]
        
        set_label = not all(element != "" for element in self.frame_label)
        return set_label
    
    

    def no_ease(self, fpt=10, istall=False, fstall=False):
        """
        Creates frame transitions with no interpolation.

        Parameters
        ----------
        fpt : integer, optional
            Number of frames per transition. The default is 10.
        istall : boolean, optional
            Determines whether to start the animation with a stall at the 
            initial stage. The default is False.
        fstall : boolean, optional
            Determines whether to end the animation with a stall at the 
            final stage. The default is False.

        Returns
        -------
        self

        """
        
        if istall:
            self.int_t = np.append(self.int_t[:1], self.int_t)
            self.labels = np.append(self.labels[:1], self.labels)
            
        if fstall:
            self.int_t = np.append(self.int_t, self.int_t[-1:])
            self.labels = np.append(self.labels, self.labels[-1:])
        
        self.n_steps = fpt
        self.n_frames = (len(self.int_t)-1) * fpt
        self._assign_main_frames()
        self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
        self.frame_label = [''] * self.n_frames
               
        
        
        set_label = True
        for z in range(np.shape(self.data)[1]): # xy coordinates
            for i in range(len(self.int_t)-1):  # states
                for frame in range(self.n_steps): # frames
                    self.eased[frame + fpt*i, z] = self.data[self.int_t[int(np.round(frame / self.n_steps)+i)], z]
                    
                if set_label: 
                    start = self.data[self.int_t[i], z]
                    end = self.data[self.int_t[i + 1], z]
                    delta = end - start
                    set_label = self._set_label(i, z, start, end, delta, fpt)

        return self


    def power_ease(self, n=3, fpt=10, istall=False, fstall=False):
        """
        Creates frame transitions based on powers (e.g. linear, quadratic, cubic etc.)

        Parameters
        ----------
        n : integer
            Exponent of the power smoothing.
        fpt : integer, optional
            Number of frames per transition. The default is 10.
        istall : boolean, optional
            Determines whether to start the animation with a stall at the 
            initial stage. The default is False.
        fstall : boolean, optional
            Determines whether to end the animation with a stall at the 
            final stage. The default is False.

        Returns
        -------
        self

        """
        
        if istall:
            self.int_t = np.append(self.int_t[:1], self.int_t)
            self.labels = np.append(self.labels[:1], self.labels)
            
        if fstall:
            self.int_t = np.append(self.int_t, self.int_t[-1:])
            self.labels = np.append(self.labels, self.labels[-1:])
        
        
        self.n_steps = fpt
        self.n_frames = (len(self.int_t)-1) * fpt
        self._assign_main_frames()
        self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
        self.frame_label = [''] * self.n_frames
        
        
        set_label = True
        sign = n % 2 * 2
        for z in range(np.shape(self.data)[1]): # xy coordinates
            
            for i in range(len(self.int_t)-1):  # states
                start = self.data[self.int_t[i], z]
                end = self.data[self.int_t[i + 1], z]
                delta = end - start
                j = 0
                    
                for t in np.linspace(0, 2, self.n_steps):
                    if (t < 1):
                        val = delta / 2 * t ** n + start

                    else:
                        t -= 2
                        val = (1 - sign) * (-delta / 2) * (t ** n - 2 * (1 - sign)) + start

                    self.eased[j + fpt*i, z] = val
                    j += 1
                    
                    
                if set_label: 
                    set_label = self._set_label(i, z, start, end, delta, fpt)


        return self


    def overshoot_ease(self, freq=0.1, fpt=30, istall=False, fstall=False):
        """
        Creates frame transitions based on overshooting via a sine function,
        which can be used to create a bouncing effect.

        Parameters
        ----------
        freq : float
            Frequency of sine function. The default is 0.1. 
        fpt : integer, optional
            Number of frames per transition. The default is 10.
        istall : boolean, optional
            Determines whether to start the animation with a stall at the 
            initial stage. The default is False.
        fstall : boolean, optional
            Determines whether to end the animation with a stall at the 
            final stage. The default is False.

        Returns
        -------
        self

        """
        
        if istall:
            self.int_t = np.append(self.int_t[:1], self.int_t)
            self.labels = np.append(self.labels[:1], self.labels)
            
        if fstall:
            self.int_t = np.append(self.int_t, self.int_t[-1:])
            self.labels = np.append(self.labels, self.labels[-1:])
            
        
        self.n_steps = fpt
        self.n_frames = (len(self.int_t)-1) * fpt
        self._assign_main_frames()
        self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
        self.frame_label = [''] * self.n_frames
        
        
        set_label = True
        amp = 0.5
        for z in range(np.shape(self.data)[1]): # xy coordinates
            
            for i in range(len(self.int_t)-1): # states
                start = self.data[self.int_t[i], z]
                end = self.data[self.int_t[i + 1], z]
                delta = end - start
                
                
                for frame in range(self.n_steps): # frames
                    
                    if delta == 0:
                        val = start
                    
                    else:
                        t = frame * 100 / fpt
                        val = delta * (1 - np.exp(-amp * freq * t)/(np.sqrt(1-amp**2)) * np.sin(np.sqrt(1-amp**2) * freq * t + np.arccos(amp))) + start
                            
                    self.eased[frame + fpt*i, z] = val
                   
                    
                if set_label:
                    set_label = self._set_label(i, z, start, end, delta, fpt)

        return self        
        
    
    def scatter_animation2d(self, duration=1.0, 
                            destination=None, save_main_frames=False, 
                            feat_kws=dict(), ax_kws=dict(), 
                            label=False, label_kws=dict(), figsize=(6, 6)):
        """
        Creates a 2d scatter plot animation.

        Parameters
        ----------
        duration : float, optional
            Duration of each stage, in seconds. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : bool or str, optional
            Determines whether keyframes are saved as PNG figures. 
            If set to a string, specifies the path where the figures 
            will be saved. The default is False.
        feat_kws : dictionary, optional
            patches.Polygon keywords. The default is an empty dictionary.
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is an empty dictionary.
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is an empty dictionary.
        figsize : tuple, optional
            Size of the plot. The default is (6,6).

        Returns
        -------
        FuncAnimation
            Animation.

        """


        # Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            raise ValueError ("Failed: Data must have an even number of columns")


        # Eased data
        it_data = self.eased

        # Set figure
        fig, ax = plt.subplots(figsize=figsize)   
        ax_kws_default={'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1, np.max(it_data)+1],}
        
        ax_kws = {**ax_kws_default, **ax_kws}
        ax.set(**ax_kws)

        
        # Set feature
        feat_kws_default={'color' : 'black',
                          'marker' : 'o',
                          'markersize' : 5,
                          'linestyle' : 'none',
                          'alpha' : 1.0}
    
        feat_kws = {**feat_kws_default, **feat_kws}


        n_dots = int(np.shape(self.data)[1]/2) # because columns has stacked x and y, so only half the size of data are points.
        dots = []
        for i in range(n_dots):
            dots.append(ax.plot([], [], **feat_kws))
            

        def animate(z):
            ax.clear()
            ax.set(**ax_kws)
            
            _anim_points(ax, z, it_data, n_dots, feat_kws)
                
            if label:
                _add_label(ax, self.frame_label[z], ax_kws, label_kws)
                 
            if save_main_frames != False and z in list(zip(*self.main_frames))[0]:
                self._save_main_frames(save_main_frames, destination, z)

            
        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=1000*duration/self.n_steps, 
                                       repeat=False, blit=False)


        if destination is not None:
            _save_anim(anim, destination, fps=self.n_steps)

        return anim


    def polyline_animation2d(self, duration=1.0, 
                             destination=None, save_main_frames=False, 
                             feat_kws=dict(), ax_kws=dict(), 
                             label=False, label_kws=dict(), figsize=(6, 6)):
        """
        Creates a 2d line plot animation.

        Parameters
        ----------
        duration : float, optional
            Duration of each stage, in seconds. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : bool or str, optional
            Determines whether keyframes are saved as PNG figures. 
            If set to a string, specifies the path where the figures 
            will be saved. The default is False.
        feat_kws : dictionary, optional
            patches.Polygon keywords. The default is dict().
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is dict().
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is dict().
        figsize : tuple, optional
            Size of the plot. The default is (6,6).

        Returns
        -------
        FuncAnimation
            Animation.

        """

        # Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            raise ValueError ("Failed: Data must have an even number of columns")


        # Eased data
        it_data = self.eased
        n_dots = int(np.shape(self.data)[1]/2) 

        # Set figure
        fig, ax = plt.subplots(figsize=figsize)   
        ax_kws_default={'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1, np.max(it_data)+1],}
        
        ax_kws = {**ax_kws_default, **ax_kws}
        

        
        # Set feature
        """
        feat_kws_default={'color' : 'black',
                          'linestyle' : '-',
                          'linewidth' : 1.0,
                          'alpha' : 1.0}
    
        feat_kws = {**feat_kws_default, **feat_kws}
        """
        
        def animate(z):
            ax.clear()
            ax.set(**ax_kws)

            _anim_line(ax, z, it_data, n_dots, feat_kws)
                
            if label:
                _add_label(ax, self.frame_label[z], ax_kws, label_kws)
                
            if save_main_frames != False and z in list(zip(*self.main_frames))[0]:
                self._save_main_frames(save_main_frames, destination, z)
        

        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=1000*duration/self.n_steps, 
                                       repeat=False, blit=False)

        if destination is not None:
            _save_anim(anim, destination, fps=self.n_steps)

        return anim
    
    
    def polygon_animation2d(self, duration=1.0, 
                            destination=None, save_main_frames=False,
                            feat_kws=dict(), ax_kws=dict(), 
                            label=False, label_kws=dict(), figsize=(6, 6)):
        """
        Creates a 2d polygon plot animation.

        Parameters
        ----------
        duration : float, optional
            Duration of each stage, in seconds. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : bool or str, optional
            Determines whether keyframes are saved as PNG figures. 
            If set to a string, specifies the path where the figures 
            will be saved. The default is False.
        feat_kws : dictionary, optional
            patches.Polygon keywords. The default is dict().
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is dict().
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is dict().
        figsize : tuple, optional
            Size of the plot. The default is (6,6).

        Returns
        -------
        FuncAnimation
            Animation.

        """

        # Running checks on data for mis-shappen arrays.
        if np.shape(self.data)[1]%2!=0:
            raise ValueError ("Failed: Data must have an even number of columns")


        # Eased data
        it_data = self.eased

        # Set figure
        fig, ax = plt.subplots(figsize=figsize)   
        ax_kws_default={'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1, np.max(it_data)+1],}
        
        ax_kws = {**ax_kws_default, **ax_kws}
        ax.set(**ax_kws)

        
        # Set feature
        feat_kws_default={'facecolor' : 'black',
                          'alpha' : 1.0}
    
        feat_kws = {**feat_kws_default, **feat_kws}


        n_dots = int(np.shape(self.data)[1]/2) # because columns has stacked x and y, so only half the size of data 
        def animate(z):
            ax.clear()
            ax.set(**ax_kws)

            _anim_polygon(ax, z, it_data, n_dots, feat_kws) 
            
            if label:
                _add_label(ax, self.frame_label[z], ax_kws, label_kws)
                
            if save_main_frames != False and z in list(zip(*self.main_frames))[0]:
                self._save_main_frames(save_main_frames, destination, z)


        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=1000*duration/self.n_steps, 
                                       repeat=False, blit=False)


        if destination is not None:
            _save_anim(anim, destination, fps=self.n_steps)

        return anim



def _anim_points(ax, z, data, n_dots, feat_kws=dict()):
    xy = np.zeros([n_dots, 2])
    for i in range(n_dots):
        xy[i,:] = data[z, i*2], data[z, i*2+1]
        
    ax.plot(xy[:,0], xy[:,1], **feat_kws)  
    
    
def _anim_line(ax, z, data, n_dots, feat_kws=dict()):
    _anim_points(ax, z, data, n_dots, feat_kws)  
    
    
def _anim_polygon(ax, z, data, n_dots, feat_kws=dict()):
    dots = []
    for i in range(n_dots):
        dots.append([data[z, i*2], data[z, i*2+1]])
        
    ax.add_patch(Polygon(dots, **feat_kws))

def _anim_arrow(ax, z, data, n_dots, feat_kws=dict()):   
    xywh = []
    for i in range(int(n_dots/2)):
        xywh.append([data[z, i*4], data[z, i*4+1], data[z, i*4+2], data[z, 1*4+3]])
        
    xywh = np.array(xywh)
    ax.quiver(xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3], **feat_kws)


def _add_label(ax, s, ax_kws, label_kws):
    label_kws_default = {"text" : s,
                         "horizontalalignment" : 'right',
                         "verticalalignment" : 'top',
                         "fontsize" : 12,
                         "position" : (ax_kws['xlim'][1]*0.95, ax_kws['ylim'][1]*0.95)}
    
    label_text = ax.text(0, 0, '')
    label_kws_ = {**label_kws_default, **label_kws}
    label_text.set(**label_kws_)


def _save_anim(anim, destination, fps):
    if destination.split('.')[-1]=='mp4':
        writer = animation.writers['ffmpeg'](fps=60)
        anim.save(destination, writer=writer, dpi=100)
        
    if destination.split('.')[-1]=='gif':
        anim.save(destination, writer='imagemagick', fps=fps, savefig_kwargs={"transparent": True})
        

def _save_main_frames(save_main_frames, destination, z, main_frames):
    idx = list(zip(*main_frames))[0].index(z)
    suffix = str(list(zip(*main_frames))[1][idx])
    if isinstance(save_main_frames, str):
        plt.savefig(save_main_frames + suffix + ".png")
    elif destination:
        path = '.'.join(destination.split('.')[:-1])
        plt.savefig(path + suffix + ".png")
    else:
        plt.savefig(suffix + ".png")
        

def animation2d(eased_list, anim_type, 
                duration=1.0, destination=None, save_main_frames=False, 
                feats_kws=[], ax_kws=dict(), axis_off=False, label=False, label_kws=dict(),
                figsize=(6,6)):
    """
    Creates a multi-feature 2d animation, from a given list of `Eased` instances.

    Parameters
    ----------
    eased_list : list
        List of `Eased` instances to animate.
    anim_type : list
        List of strings representing the desired the animation 
        type for each eased instance. Options are points, line 
        and polygon.
    duration : float, optional
        Duration of each stage, in seconds. The default is 1.0. 
    destination : string, optional
        Output path for the animation. The default is None.
    save_main_frames : bool or str, optional
        Determines whether keyframes are saved as PNG figures. 
        If set to a string, specifies the path where the figures 
        will be saved. The default is False.
    feats_kws : list of dicts, optional
        List of patches.Polygon keywords. The default is [].
    ax_kws : dict, optional
        Matplotlib.Axes keywords. The default is dict().
    axis_off : bool, optional
        Determines whether to turn off the axes. If set to False, 
        axes are plotted. The default is False.
    label : bool, optional
        True for plotting labels in the plot. Labels can only be taken from
        the keys of the dictionary passed as data. The default is False.
    label_kws : dict, optional
        Matplotlib.Text keywords. The default is. The default is dict().
    figsize : tuple, optional
        Size of the plot. The default is (6,6).
    """


    #Running checks on data for mishappen arrays.
    for eased in eased_list:
        if np.shape(eased.data)[1]%2!=0:
            raise ValueError ("Failed: Elements in eased_list must have an even number of columns")

        
    if len(eased_list) != len(anim_type):
        raise ValueError ("Failed: eased_list and anim_list must have the same length")
        
    n_frames_list = [eased.n_frames for eased in eased_list]
    if len(set(n_frames_list)) != 1:
        print('\033[91m' + "Warning: N_frames accross Eased() instances is not the same length")
        
    n_steps_list = [eased.n_steps for eased in eased_list]
    if len(set(n_steps_list)) != 1:
        print('\033[91m' + "Warning: N_steps accross Eased() instances is not the same length")
        
        
    n_feat = len(eased_list)
    
    
    # Set figure
    fig, ax = plt.subplots(figsize=figsize)   
    xlims = np.zeros([n_feat,2]) 
    ylims = np.zeros([n_feat,2]) 
    
    if not feats_kws:
        feats_kws = [dict()] * len(eased_list)
    
    for i, eased_ in enumerate(eased_list):
        
        # Eased data
        it_data = eased_.eased

        
        xlims[i, :] = np.min(it_data)-1, np.max(it_data)+1
        ylims[i, :] = np.min(it_data)-1, np.max(it_data)+1
        
        # Set feature
        if anim_type[i] == "points":
            feat_kws_default={'color' : 'black',
                              'marker' : 'o',
                              'markersize' : 5,
                              'linestyle' : 'none',
                              'alpha' : 1.0}
            
        elif anim_type[i] == "line":
            feat_kws_default={'color' : 'black',
                              'linestyle' : '-',
                              'linewidth' : 1.0,
                              'alpha' : 1.0}
        
        elif anim_type[i] == "polygon":
            feat_kws_default={'facecolor' : 'black',
                              'alpha' : 0.7}
            
        elif anim_type[i] == "arrow":
            feat_kws_default={'facecolor' : 'black',
                              'alpha' : 0.7}
            
        else:
            raise TypeError ("Failed: Unrecognised animation type, options are points, line or polygon")
            
        
        if feats_kws:
            if len(feats_kws) != len(eased_list):
                raise ValueError ("Failed: If given, feats_kws must have the same length as eased_list")
            else:
                feats_kws[i] = {**feat_kws_default, **feats_kws[i]}

    

    ax_kws_default={'xlim':[np.min(xlims[:,0])-1, np.max(xlims[:,1])+1],
                    'ylim':[np.min(ylims[:,0])-1, np.max(ylims[:,1])+1]}
    
    ax_kws = {**ax_kws_default, **ax_kws}


    n_dots_list = [int(np.shape(eased_.data)[1]/2) for eased_ in eased_list]
    

    def animate(z):
        ax.clear()
        ax.set(**ax_kws)
        if axis_off:
            ax.set_axis_off()

        for i, eased_ in enumerate(eased_list):
            
            it_data = eased_.eased
            
            if anim_type[i] == "points":
                _anim_points(ax, z, it_data, n_dots_list[i], 
                             feat_kws=feats_kws[i])

                
            if anim_type[i] == "line":
                _anim_line(ax, z, it_data, n_dots_list[i], 
                           feat_kws=feats_kws[i])                
            
            elif anim_type[i] == "polygon":
                _anim_polygon(ax, z, it_data, n_dots_list[i], 
                              feat_kws=feats_kws[i]) 
                
            elif anim_type[i] == "arrow":
                _anim_arrow(ax, z, it_data, n_dots_list[i], 
                            feat_kws=feats_kws[i]) 
        
            
        if label:
            _add_label(ax, eased_list[0].frame_label[z], ax_kws, label_kws)
            
            
        if save_main_frames and z in list(zip(*eased_list[0].main_frames))[0]:
            _save_main_frames(save_main_frames, destination, z, eased_list[0].main_frames)


        

    anim = animation.FuncAnimation(fig, animate, frames=eased_list[0].n_frames, 
                                   interval=1000*duration/eased_list[0].n_steps, 
                                   repeat=False, blit=False)

    if destination is not None:
        _save_anim(anim, destination, fps=eased_list[0].n_steps)

    return anim
    

if __name__ == "__main__":
    
    print('EASING : A library for smooth animations in python : version 0.1.0')