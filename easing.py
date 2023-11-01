import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, rc
rc('animation', html='html5')
from IPython.display import HTML, Image
from itertools import groupby
from matplotlib.patches import Polygon


class Eased:
    """ This class takes the original time vector and raw data (as a m*n 
    matrix or DataFrame) along with an output vector and interpolation function
    For the input data, the rows are the different variables and the columns 
    correspond to the time points"""

    def __init__(self, data, in_t=None, wrap=True, istall=False, fstall=False):
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
        istall : boolean, optional
            Tells the easer whether to start the animation with a stall at the 
            initial state. The default is False.
        fstall : boolean, optional
            Tells the easer whether to end the animation with a stall at the 
            final state. The default is True.
            
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
            
        if istall:
            in_t = np.append(in_t[:1], in_t)
            labels = np.append(labels[:1], labels)
            
        if fstall:
            in_t = np.append(in_t, in_t[-1:])
            labels = np.append(labels, labels[-1:])
            
            
        self.labels = labels
        self.n_dims = len(np.shape(data))
        self.int_t = in_t
        self.data = data
        

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

        return self.eased


    def power_ease(self, n, fpt=10):
        """
        Takes care of interpolating (easing) the coordinates according to the 
        given fpt.

        Parameters
        ----------
        n : integer
            Exponent of the power smoothing.
        fpt : integer, optional
            Number of frames per transition. The default is 10.

        Returns
        -------
        Array of floats
            An array of floats with states(frames) as rows and x- and y-
            coordinates for each point as columns.

        """
        
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
                    j += 1
            self.eased[j:] = self.data[i + 1]

        else:
            self.eased = np.zeros( (self.n_frames, np.shape(self.data)[1]) )
            for z in range(np.shape(self.data)[1]): 
                j = 0
                
                for i in range(len(self.int_t)-1): 
                    start = self.data[self.int_t[i], z]
                    end = self.data[self.int_t[i + 1], z]
                    
                    for t in np.linspace(0, 2, self.n_steps):
                        if (t < 1):
                            val = (end - start) / 2 * t ** n + start

                        else:
                            t -= 2
                            val = (1 - sign) * (-(end - start) / 2) * (t ** n - 2 * (1 - sign)) + start

                        self.eased[j, z] = val
                        j += 1


        return self.eased


    def scatter_animation2d(self, n=3, fpt=30, speed=1.0, gif=False, destination=None,plot_kws=None,label=False):
        """
        Flexibly create a 2d scatter plot animation.

        This function creates a matplotlib animation from a pandas Dataframe or a MxN numpy array. The Columns are paired
        with x and y coordinates while the rows are the individual time points.

        If a DataFrame is passed, the data columns are expected to have the xy values for each point stacked in pairs.
        You would get that from e.g.: w=np.random.multivariate_normal([1,1],[[4, 2], [2, 4]],size=size).reshape(1,-1)
        where sampling is done for both axis in the same call.
        
        This takes a number of parameters for the animation, as well as


        Parameters
        ----------
        n: Exponent of the power smoothing
        fpt: how smooth the frames of the animation are
        speed: speed
        inline:
        gif:
        destination:
        :return:
        """


        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)") 


        if plot_kws is None:
            plot_kws = dict()


        it_data = self.power_ease(n,fpt)

        # filling out missing keys
        vanilla_params={'s':10,'color':'black','xlim':[np.min(it_data)-1, np.max(it_data)+1],'ylim':[np.min(it_data)-1,np.max(it_data)+1],'xlabel':'','ylabel':'','alpha':1.0,'figsize':(6,6)}
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]



        fig, ax = plt.subplots(figsize=plot_kws['figsize'])
        ax.set_xlim(plot_kws['xlim'])
        ax.set_ylim(plot_kws['ylim'])
        ax.set_xlabel(plot_kws['xlabel'])
        ax.set_ylabel(plot_kws['ylabel'])

        if label==True:
            label_text = ax.text(plot_kws['xlim'][1]*0.75, plot_kws['ylim'][1]*.9, '',fontsize=18)

        n_dots=int(np.shape(self.data)[1]/2) # because columns has stacked x and y, so only half the size of data are points.
        dots=[]
        for i in range(n_dots):
            dots.append(ax.plot([], [], linestyle='none', marker='o', markersize=plot_kws['s'], color=plot_kws['color'], alpha=plot_kws['alpha']))



        def animate(z):
            for i in range(n_dots):
                dots[i][0].set_data(it_data[z,i*2],it_data[z,i*2+1])
            if label==True:
                label_text.set_text(self.labels[int(np.floor((z+fpt/2)/fpt))])
                return dots,label_text
            else:
                return dots

        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames,interval=400/fpt/speed, blit=False)


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


    def polygon_animation2d(self, n=3, fpt=30, speed=1.0, ease_method='power_ease',
                            gif=False, destination=None, plot_kws=None, label=False):
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
        n : integer
            Exponent of the power smoothing. The default is 3.
        fpt : integer, optional
            Number of frames per transition. The default is 30.
        speed : float, optional
            DESCRIPTION. The default is 1.0.
        ease_method : string, optional
            Smooting method to be used. The default is 'power_ease'.
        gif : boolean, optional
            DESCRIPTION. The default is False.
        destination : string, optional
            Output path for the animation. The default is None.
        plot_kws : dictionary, optional
            Plotting keywords. The default is None.
        label : boolean, optional
            True for plotting labels in the plot. Labels can only be taken from
            the keys of the dictionary passed as data. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        #Running checks on data for mishappen arrays.
        if np.shape(self.data)[1]%2!=0:
            print('\033[91m' + "Failed: Data must have an even number of columns")
            exit()
        if np.shape(self.data)[0]<np.shape(self.data)[1]:
            print('\033[91m' + "Warning : Data has more columns (xys) than rows (time)") # !!! when is this relevant


        if plot_kws is None:
            plot_kws = dict()


        # Ease data
        if ease_method == "power_ease":
            it_data = self.power_ease(n, fpt)
            

        # Fill missing plotting keywords (otherwise given in plot_kws)
        # !!! update with concerning keywords
        # !!! Should perhaps divide into plot kws and polygon kws.
        vanilla_params={'color':'black',
                        'xlim':[np.min(it_data)-1, np.max(it_data)+1],
                        'ylim':[np.min(it_data)-1,np.max(it_data)+1],
                        'xlabel':'',
                        'ylabel':'',
                        'alpha':1.0,
                        'figsize':(6,6),
                        }
        
        for key in vanilla_params.keys():
            if key not in plot_kws.keys():
                plot_kws[key] = vanilla_params[key]


        # Set figure
        fig, ax = plt.subplots(figsize=plot_kws['figsize'])
        ax.set_xlim(plot_kws['xlim'])
        ax.set_ylim(plot_kws['ylim'])
        ax.set_xlabel(plot_kws['xlabel'])
        ax.set_ylabel(plot_kws['ylabel'])


        if label==True:
            label_text = ax.text(plot_kws['xlim'][1]*0.75, plot_kws['ylim'][1]*.9, '',fontsize=18) # set later in the code with "set_text")

        
        n_dots=int(np.shape(self.data)[1]/2) # because columns has stacked x and y, so only half the size of data 
        poly = ax.add_patch(Polygon([[0,0]], fc=plot_kws['color'], alpha=plot_kws['alpha']))
     

        def animate(z):
            dots = []
            for i in range(n_dots):
                dots.append([it_data[z,i*2], it_data[z,i*2+1]])
                
            poly.set_xy(dots)
            
            if label==True:
                label_text.set_text(self.labels[int(np.floor((z+fpt/2)/fpt))])  #!!! This will only work for DataFrames I think
                return poly,label_text
            else:
                return poly

        anim = animation.FuncAnimation(fig, animate, frames=self.n_frames, 
                                       interval=400/fpt/speed, blit=False)


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



    def barchart_animation(self,n=3,fpt=30,speed=1.0,gif=False,destination=None,plot_kws=None,label=False,zero_edges=True,loop=True):
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

    def timeseries_animation(self,n=1,speed=1.0,interp_freq=0,starting_pos = 25,gif=False,destination=None,plot_kws=None,final_dist=False):
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