import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, rc
rc('animation', html='html5')
from IPython.display import HTML, Image
from itertools import groupby
from matplotlib.patches import Polygon
import math


class Eased:
    """ This class takes the original time vector and raw data (as a m*n 
    matrix or DataFrame) along with an output vector and interpolation function
    For the input data, the rows are the different variables and the columns 
    correspond to the time points"""

    def __init__(self, data, in_t=None, wrap=True):
        """
        Create a new instance of the class Eased().

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        in_t : TYPE, optional
            DESCRIPTION. The default is None.
        wrap : boolean, optional
            Tells the easer whether wrap the animation back to the intial state. 
            The default is True.
            
        Returns
        -------
        None.

        """
        
        if isinstance(data, pd.DataFrame):
            
            self.columns = data.columns
            
            if in_t is None:
                in_t = np.arange(len(data.index.values))     
            
            
            labels = data.index.values[in_t]
            data = data.values
            
        
        elif isinstance(data, np.ndarray):
            
            if in_t is None:
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
        
    # Not mantained
    def no_ease(self, fpt=10): # not sure how to replace fpt and keep the time stretch
        """
        Maps the input vector over the outuput time vector without interpolation.

        Parameters
        ----------
        fpt : integer, optional
            Number of frames per transition. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        n_frames = len(self.int_t)*fpt
        self.n_frames = n_frames
        self.n_steps = int( np.ceil(self.n_frames / len(self.int_t)) )


        if self.n_dims == 1: # if the input is only one row
            self.eased = np.zeros((self.n_frames, 1))
            
            for i, t in enumerate(self.n_frames):
                self.eased[i] = self.data[int(np.floor(i / self.n_steps))]
                
        else: # if the input is a multidimensional row
            self.eased = np.zeros((np.shape(self.data)[0], self.n_frames))
            
            for z in range(np.shape(self.data)[0]):
                
                for i, t in enumerate(self.n_frames):
                    self.eased[z, i] = self.data[z, int(np.floor(i / self.n_steps))]

        return self


    def power_ease(self, n, fpt=10, istall=False, fstall=False):
        """
        Takes care of interpolating (easing) the coordinates according to the 
        given fpt.

        Parameters
        ----------
        n : integer
            Exponent of the power smoothing.
        fpt : integer, optional
            Number of frames per transition. The default is 10.
        istall : boolean, optional
            Tells the easer whether to start the animation with a stall at the 
            initial state. The default is False.
        fstall : boolean, optional
            Tells the easer whether to end the animation with a stall at the 
            final state. The default is True.

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
        
        n_frames = (len(self.int_t)-1) * fpt
        self.n_frames = n_frames
        self.n_steps = int( np.ceil(self.n_frames / (len(self.int_t)-1)) )
        
        
        sign = n % 2 * 2
        if self.n_dims == 1:
            self.eased = np.zeros((self.n_frames, 1))
            j = 0
            for i in range(len(self.int_t) - 1):

                start = self.data[i]
                end = self.data[i + 1]
                for t in np.linspace(0, 2, self.n_steps):
                    if (t < 1):
                        val = (end - start) / 2 * t ** n + start

                    else:
                        t -= 2
                        val = (1 - sign) * (-(end - start) / 2) * (t ** n - 2 * (1 - sign)) + start

                    self.eased[j] = val

            self.eased[j:] = self.data[i + 1]

        else:
            self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
            self.frame_label = [''] * self.n_frames
            set_label = True
            
            
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

                        self.eased[j, z] = val
                        j += 1
                        
                    if set_label: # Set state labels based on coordinate position
                        
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

        return self


    def overshoot_ease(self, freq=0.1, fpt=30, istall=False, fstall=False):
        """
        Takes care of interpolating (easing) the coordinates according to the 
        given fpt.

        Parameters
        ----------
        freq : float
            Frequency of sine function. The default is 0.1. 
        fpt : integer, optional
            Number of frames per transition. The default is 10.
        istall : boolean, optional
            Tells the easer whether to start the animation with a stall at the 
            initial state. The default is False.
        fstall : boolean, optional
            Tells the easer whether to end the animation with a stall at the 
            final state. The default is True.

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
            
        amp = 0.5
        n_frames = (len(self.int_t)-1) * fpt
        self.n_frames = n_frames
        self.n_steps = int( np.ceil(self.n_frames / (len(self.int_t)-1)) )
        
        
        # Save main frames
        states = np.unique(self.int_t)
        self.main_frames = []
        for state in states:
            idx = np.where(self.int_t == state)[0][0]
            self.main_frames.append(self.n_steps * idx)
        
        # Erased cade for n_dims == 1. Not sure the purpose.
        
        self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
        self.frame_label = [''] * self.n_frames
        set_label = True
        
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

                   
                    
                if set_label: # Set state labels based on coordinate position
                    
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

        return self        
        
        

    def scatter_animation2d(self, speed=1.0, 
                            destination=None, save_main_frames=False, 
                            feat_kws=dict(), ax_kws=dict(), 
                            label=False, label_kws=dict()):
        """
        Create a 2d scatter plot animation.

        This function creates a matplotlib animation from a pandas Dataframe 
        or a MxN numpy array. The Columns are paired with x- and y-coordinates 
        while the rows are the individual time points.

        If a DataFrame is passed, the data columns are expected to have the xy 
        values for each point stacked in pairs. You would get that from e.g.: 
        w = np.random.multivariate_normal([1,1],[[4, 2], [2, 4]],size=size).reshape(1,-1)
        where sampling is done for both axis in the same call.


        Parameters
        ----------
        speed : float, optional
            DESCRIPTION. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : TYPE, optional
            DESCRIPTION. The default is False.
        feat_kws : dictionary, optional
            Mpatches.Polygon keywords. The default is an empty dictionary.
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is an empty dictionary.
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is an empty dictionary.

        Returns
        -------
        FuncAnimation
            Animation.

        """


        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)") 


        # Eased data
        it_data = self.eased

        # Set figure
        fig, ax = plt.subplots()   
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
            
            
        if label:
            label_text = ax.text(0, 0, '')


        def animate(z):
            for i in range(n_dots):
                dots[i][0].set_data(it_data[z,i*2],it_data[z,i*2+1])
                
            if label:
                label_kws = {}
                label_kws_default = {"text" : self.frame_label[z],
                                     "horizontalalignment" : 'right',
                                     "verticalalignment" : 'top',
                                     "fontsize" : 18,
                                     "position" : (ax_kws['xlim'][1]*0.95, ax_kws['ylim'][1]*0.95)}
                
                label_kws = {**label_kws_default, **label_kws}
                label_text.set(**label_kws)
                 
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                        
                return dots, label_text
            
            else:
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                        
                return dots

        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=400/self.n_steps/speed, 
                                       repeat=False, blit=False)


        if destination is not None:
            
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
                
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=self.n_steps)

        return anim


    def polyline_animation2d(self, speed=1.0, 
                             destination=None, save_main_frames=False, 
                             feat_kws=dict(), ax_kws=dict(), 
                             label=False, label_kws=dict()):
        """
        Create a 2d line plot animation.

        This function creates a matplotlib animation from a pandas Dataframe 
        or a MxN numpy array. The Columns are paired with x- and y-coordinates 
        while the rows are the individual time points.

        If a DataFrame is passed, the data columns are expected to have the xy 
        values for each point stacked in pairs. You would get that from e.g.: 
        w = np.random.multivariate_normal([1,1],[[4, 2], [2, 4]],size=size).reshape(1,-1)
        where sampling is done for both axis in the same call.


        Parameters
        ----------
        speed : float, optional
            DESCRIPTION. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : TYPE, optional
            DESCRIPTION. The default is False.
        feat_kws : dictionary, optional
            Mpatches.Polygon keywords. The default is an empty dictionary.
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is an empty dictionary.
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is an empty dictionary.

        Returns
        -------
        FuncAnimation
            Animation.

        """


        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)") 


        # Eased data
        it_data = self.eased
        n_dots = int(np.shape(self.data)[1]/2) 

        # Set figure
        fig, ax = plt.subplots()   
        ax_kws_default={'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1, np.max(it_data)+1],}
        
        ax_kws = {**ax_kws_default, **ax_kws}
        ax.set(**ax_kws)

        
        # Set feature
        feat_kws_default={'color' : 'black',
                          'linestyle' : '-',
                          'linewidth' : 1.0,
                          'alpha' : 1.0}
    
        feat_kws = {**feat_kws_default, **feat_kws}
        #lines = ax.plot(np.empty((0, y.shape[1])), np.empty((0, y.shape[1])), **feat_kws)
        lines = ax.plot([], [], **feat_kws)
            
            
        if label:
            label_text = ax.text(0, 0, '')


        def animate(z):
            
            for line in lines:
                x = np.zeros(n_dots)
                y = np.zeros(n_dots)
                for i in range(n_dots):
                    x[i] = it_data[z, i*2]
                    y[i] = it_data[z, i*2+1]
                    
                line.set_data(x, y)
            
            
                
            if label:
                label_kws = {}
                label_kws_default = {"text" : self.frame_label[z],
                                     "horizontalalignment" : 'right',
                                     "verticalalignment" : 'top',
                                     "fontsize" : 18,
                                     "position" : (ax_kws['xlim'][1]*0.95, ax_kws['ylim'][1]*0.95)}
                
                label_kws = {**label_kws_default, **label_kws}
                label_text.set(**label_kws)
                 
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                        
                return lines, label_text
            
            else:
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                        
                return lines

        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=400/self.n_steps/speed, 
                                       repeat=False, blit=False)


        if destination is not None:
            
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
                
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=self.n_steps)

        return anim
    
    def polygon_animation2d(self, speed=1.0, 
                            destination=None, save_main_frames=False,
                            feat_kws=dict(), ax_kws=dict(), 
                            label=False, label_kws=dict()):
        """
        Create a 2d polygon plot animation.

        This function creates a matplotlib animation from a pandas Dataframe or 
        a MxN numpy array. The columns are paired with x- and y-coordinates 
        while the rows are the individual time points.
        
        If a DataFrame is passed, the data columns are expected to have the xy 
        values for each point stacked in pairs. You would get that from e.g.: 
        w = np.random.multivariate_normal([1,1],[[4, 2], [2, 4]],size=size).reshape(1,-1)
        where sampling is done for both axis in the same call.


        Parameters
        ----------
        speed : float, optional
            DESCRIPTION. The default is 1.0.
        destination : string, optional
            Output path for the animation. The default is None.
        save_main_frames : boolean, optional
            Not False for saving as png the main and unique frames. This 
            parameter may be set up as a string with the figure path to be used.
            The default is False.
        feat_kws : dictionary, optional
            Mpatches.Polygon keywords. The default is an empty dictionary.
        ax_kws : dictionary, optional
            Matplotlib.Axes keywords. The default is an empty dictionary.
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.
        label_kws : dictionary, optional
            Matplotlib.Text keywords. The default is an empty dictionary.

        Returns
        -------
        FuncAnimation
            Animation.

        """

        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)") # !!! when is this relevant


        # Eased data
        it_data = self.eased

        # Set figure
        fig, ax = plt.subplots()   
        ax_kws_default={'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1, np.max(it_data)+1],}
        
        ax_kws = {**ax_kws_default, **ax_kws}
        ax.set(**ax_kws)

        
        # Set feature
        feat_kws_default={'facecolor' : 'black',
                          'alpha' : 1.0}
    
        feat_kws = {**feat_kws_default, **feat_kws}
        poly = ax.add_patch(Polygon([[0,0]], **feat_kws))
        
        if label:
            label_text = ax.text(0, 0, '')
        


        n_dots = int(np.shape(self.data)[1]/2) # because columns has stacked x and y, so only half the size of data 
        def animate(z):
            dots = []
            for i in range(n_dots):
                dots.append([it_data[z,i*2], it_data[z,i*2+1]])
                
            poly.set_xy(dots)
            
            if label:
                label_kws = {}
                label_kws_default = {"text" : self.frame_label[z],
                                     "horizontalalignment" : 'right',
                                     "verticalalignment" : 'top',
                                     "fontsize" : 18,
                                     "position" : (ax_kws['xlim'][1]*0.95, ax_kws['ylim'][1]*0.95)}
                
                label_kws = {**label_kws_default, **label_kws}
                label_text.set(**label_kws)
                
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                    
                return poly, label_text
            
            else:
                if save_main_frames != False and z in self.main_frames:
                    idx = np.where(self.main_frames == z)[0][0]
                    
                    if isinstance(save_main_frames, str):
                        plt.savefig(save_main_frames + str(idx) + ".png")
                    else:
                        plt.savefig(str(idx)+".png")
                        
                return poly


        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=400/self.n_steps/speed, 
                                       repeat=False, blit=False)


        if destination is not None:
            
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
                
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=self.n_steps)

        return anim


    # Not mantained
    def barchart_animation(self, n=3, fpt=30, speed=1.0, gif=False, 
                           destination=None, plot_kws=None, label=False, 
                           zero_edges=True, loop=True):
        '''
        This barchart animation create line barcharts that morph over time using the eased data class

        It takes the following additional arguments
        :param n: this is the power curve modifier to passed to power_ease
        :param fpt: this is a rendering parameter that determines the relative framerate over the animation
        :param speed: How quickly does the animation unfold // a value of 1 indicates the default [R>0]

        :param destination: This is the output file (if none it will be displayed inline for jupyter notebooks) - extension determines filetype


        :param plot_kws: These are the matplotlib key work arghuments that can be passed in the event the defaults don't work great
        :param label: This is an optional paramter that will display labels of the pandas rows as the animation cycles through

        :return: rendered animation
        '''



        it_data = self.power_ease(n, fpt)

        x_vect=np.arange(len(self.columns))


        ### running checks on the paramters

        #Runing checks on parameters
        assert speed>0, "Speed value must be greater than zero"


        # filling out missing keys
        vanilla_params = {'s': 10, 'color': 'black', 'xlim': [min(x_vect) - 1, max(x_vect) + 1],
                          'ylim': [np.min(it_data) - 1, np.max(it_data) + 1], 'xlabel': '', 'ylabel': '','title': '',
                          'alpha': 1.0, 'figsize': (6, 6)}
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]

        fig, ax = plt.subplots(figsize=plot_kws['figsize'])
        ax.set_xlim(plot_kws['xlim'])
        ax.set_ylim(plot_kws['ylim'])
        ax.set_title(plot_kws['title'])
        ax.set_xlabel(plot_kws['xlabel'])
        ax.set_ylabel(plot_kws['ylabel'])
        ax.set_xticks(x_vect-np.mean(np.diff(x_vect))/2)
        ax.set_xticklabels(list(self.columns),rotation=90)

        plt.tight_layout()
        if label == True:
            label_text = ax.text(plot_kws['xlim'][1] * 0.25, plot_kws['ylim'][1] * .9, '', fontsize=18)

        lines=[]
        lines.append(ax.plot([], [], linewidth=3, drawstyle='steps-pre', color=plot_kws['color'], alpha=plot_kws['alpha']))


        # add zero padding to the data // makes for prettier histogram presentation
        if zero_edges==True:
            zero_pad=np.zeros((it_data.shape[0],1))
            it_data=np.hstack((zero_pad,it_data,zero_pad))
            x_vect=[min(x_vect)-1]+list(x_vect)+[max(x_vect)+1]

        def animate(z):
            lines[0][0].set_data(x_vect, it_data[z, :])

            if label==True:
                label_text.set_text(self.labels[int(np.floor((z+fpt/2)/fpt))])
                return lines,label_text
            else:
                return lines


        anim = animation.FuncAnimation(fig, animate, frames=it_data.shape[0],interval=400/fpt/speed, blit=False)


        if destination is not None:
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=fpt)

        if gif==True:
            return Image(url='animation.gif')
        else:
            return anim


    # Not mantained
    def timeseries_animation(self, n=1, speed=1.0, interp_freq=0, 
                             starting_pos = 25, gif=False, destination=None,
                             plot_kws=None, final_dist=False):
        '''
        This method creates a timeseiers animation of ergodic processes
        :param fpt:
        :param speed:
        :param interp_freq: This is the number of steps between each given datapoint interp_freq=1 // no additional steps
        :param gif:
        :param destination:
        :param plot_kws:
        :param label:
        :param zero_edges:
        :param loop:
        :return:
        '''
        interp_freq+=1

        data = self.power_ease(n=n, fpt=interp_freq)

        assert min(data.shape)==1, "timeseries animation only take 1 dimensional arrays"

        data=[k for k, g in groupby(list(data))]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4),gridspec_kw={'width_ratios': [3, 1]},sharey=True)



        max_steps=len(data)


        vanilla_params = {'s': 10, 'color': 'black', 'xlim': [0, starting_pos],
                          'ylim': [np.min(data) - 1, np.max(data) + 1], 'xlabel': '', 'ylabel': '','title': '',
                          'alpha': 1.0, 'figsize': (12, 3),'linestyle':'none','marker':'o'}
        if plot_kws==None:
            plot_kws={}
        x_vect=np.linspace(1,starting_pos,starting_pos*interp_freq)

        # Creating NaN padding at the end for time series plot
        data = np.append(data, x_vect * np.nan)

        # fill out parameters
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]

        ax[0].set_ylim(plot_kws['ylim'])
        ax[1].set_ylim(plot_kws['ylim'])

        ax[0].set_xlim(plot_kws['xlim'])
        lines=[]
        lines.append(ax[0].plot([], [], linewidth=3, color=plot_kws['color'], alpha=plot_kws['alpha'],linestyle=plot_kws['linestyle'], marker=plot_kws['marker']))
        if 'bins' not in plot_kws.keys():
            plot_kws['bins']=np.linspace(plot_kws['ylim'][0],plot_kws['ylim'][1],20)


        #plotting light grey final dist:
        if final_dist==True:
            bins, x = np.histogram(data,bins=plot_kws['bins'])
            ax[1].plot(bins, x[1:], linewidth=3, drawstyle='steps-pre', color='#d3d3d3')

        else:
            bins, x = np.histogram(data,bins=plot_kws['bins'])
            ax[1].plot(bins, x[1:], linewidth=3, drawstyle='steps-pre', color='#d3d3d3',alpha=0)


        histlines=[]
        histlines.append(ax[1].plot([], [], linewidth=3,  drawstyle='steps-pre',color=plot_kws['color'], alpha=plot_kws['alpha']))


        # This function plots the distribution of flowing information // so we start at the beining and plot forward
        # reverse the orientation of data
        trace_data=data[::-1]


        def animate(z):
            lines[0][0].set_data(x_vect, trace_data[-(starting_pos*interp_freq+1)-z:-1-z])

            # compute the histogram of what what has passed
            if z>0:
                bins, x = np.histogram(trace_data[-(z):-1],bins=plot_kws['bins'])
                histlines[0][0].set_data(bins,x[1:])
                lines.append(ax[1].plot([], [], linewidth=3, color=plot_kws['color'], alpha=plot_kws['alpha']))


            return lines


        anim = animation.FuncAnimation(fig, animate, frames=max_steps,interval=400/speed, blit=False)


        if destination is not None:
            if destination.split('.')[-1]=='mp4':
                writer = animation.writers['ffmpeg'](fps=60)
                anim.save(destination, writer=writer, dpi=100)
            if destination.split('.')[-1]=='gif':
                anim.save(destination, writer='imagemagick', fps=30)

        if gif==True:
            return Image(url='animation.gif')
        else:
            return anim



if __name__ == "__main__":
    
    print('EASING : A library for smooth animations in python : version 0.1.0')