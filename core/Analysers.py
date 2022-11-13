from core.Processors import Processor
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

from core.Tools import get_figure
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,TextArea)
from core.Tools import get_figure
from sklearn.utils import shuffle

#font = {'size'   : 12}

#import matplotlib
#matplotlib.rc('font', **font)


class Analyser(Processor):
    
    type = "analyser"

    def process(self,input_data):
        self.analyse(np.ravel(input_data))
        return input_data

    def analyse(self,input_data):
        pass

class Printer(Analyser):
    
    def __init__(self, N=10,name = "Printer",type=None):
        self.N = N
        self.name = name
        self.type = type

    def analyse(self,input_data):
        print("--- Printer {} ---".format(self.name))
        if verbose=="True":
            if input_data.ndim == 2:
                print(input_data[:,:self.N])
            else:
                print(input_data[:self.N])

class Scope(Analyser):

    """ Scope
        
        This object can be used to display data
        
        """

    def __init__(self, type="plot", fs=1, figure= None, options=None, dual_mode=False, name = "figure name", **kwargs):
        self.type = type
        self.figure = figure
        self.name = name
        self.fs = fs
        self.options = options
        self.dual_mode = dual_mode
        self.kwargs = kwargs

    def get_error_ellipse(self, centre, conf, cov, **kwargs):
        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by 
        vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * 5 * np.sqrt(conf*eigvals)
        return Ellipse(xy=centre, width=width, height=height,angle=np.degrees(theta), fc='r')

    def get_plot_function(self):
        if (self.type == "scatter") or (self.type == "plot"):
            plot_function = plt.plot

        if self.type == "stem":
            plot_function = plt.stem

        return plot_function

    def analyse(self,input_data):
        
        if self.kwargs.values() and list(self.kwargs.values())[0]==True:
            cov = getattr(list(self.kwargs.values())[1],'Cov')

            fig, ax = plt.subplots(figsize=(19,19))
            box1 = TextArea("    95% Error Ellipse : ", textprops=dict(color="k"))

            box2 = DrawingArea(60, 80, 25, 40)
            el = self.get_error_ellipse((0,0),list(self.kwargs.values())[2],cov)
            box2.add_artist(el)

            box = HPacker(children=[box1, box2],align="center",pad=0, sep=5)

            anchored_box = AnchoredOffsetbox(loc='lower left',
                                 child=box, pad=0.,
                                 frameon=True,
                                 bbox_to_anchor=(0., 1.02),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )

            ax.add_artist(anchored_box)

            fig.subplots_adjust(top=0.8)
        else:
            fig = get_figure(figure_name=self.figure)
        
        if self.dual_mode == False:
            plot_function = self.get_plot_function()
        else:
            plot_function1 = self.get_plot_function()
            plot_function2 = self.get_plot_function()
        
        if self.type == "scatter":
            # extract maximum of absolute values in the two complex axes
            if self.dual_mode == False:
                plot_function(np.real(input_data),np.imag(input_data),'.', markersize=9)
                plt.xlabel("Real Part")
                plt.ylabel("Imaginary Part")
                plt.axis('scaled')
            else:
                N = len(input_data)//2
                in1 = input_data[:N]
                in2 = input_data[N:]
                plot_function1(np.real(in1),np.imag(in1),'b.', markersize=9)
                plt.title("First Polarization")
                plt.xlabel("Real Part")
                plt.ylabel("Imaginary Part")
                plt.axis('scaled')
                plt.figure()
                plot_function2(np.real(in2),np.imag(in2),'r.', markersize=9)
                plt.title("Second Polarization")
                plt.xlabel("Real Part")
                plt.ylabel("Imaginary Part")
                plt.axis('scaled')

        
        else:

            if input_data.shape[0] > 1:
                # convert block data to stream data
                input_data = np.ravel(input_data,order="F")

            t = np.arange(len(input_data))/self.fs

            if np.iscomplexobj(input_data):
                
                plt.subplot(2,1,1)
                plot_function(t, np.real(input_data))
                plt.xlabel("time (s)")
                plt.ylabel("real part")
        
                plt.subplot(2,1,2)
                plot_function(t, np.imag(input_data))
                plt.xlabel("time (s)")
                plt.ylabel("imaginary part")
    
            else:
                plot_function(t, input_data)
                plt.xlabel("time (s)")


class Spectral_Scope(Analyser):

    """ Spectral Scope
        
        This object can be used to display data
        
    """

    def __init__(self, type ="plot", figure_mag= None, figure_phase=None, phase=False, fs= 1.0, fc=0., scale='dB', nfft=None, dual_mode=False, name = "figure name"):
        self.type = type
        self.fs = fs
        self.fc = fc
        self.phase = phase
        self.scale = scale
        self.nfft = nfft
        self.figure_mag = figure_mag
        self.figure_phase = figure_phase
        self.dual_mode = dual_mode
        self.name = name

    def analyse(self,input_data):
        if self.dual_mode == False: 
            figure_mag = get_figure(figure_name=self.figure_mag)
            plt.magnitude_spectrum(x=input_data, Fs=self.fs, Fc=self.fc, scale=self.scale)

            if self.phase:
                figure_phase = get_figure(figure_name=self.figure_phase)
                plt.phase_spectrum(x=input_data, Fs=self.fs, Fc=self.fc)
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            figure_mag = get_figure(figure_name="Magnitude spectrum - Pol1")
            plt.magnitude_spectrum(x=in1, Fs=self.fs, Fc=self.fc, scale=self.scale)
            figure_mag = get_figure(figure_name="Magnitude spectrum - Pol2")
            plt.magnitude_spectrum(x=in2, Fs=self.fs, Fc=self.fc, scale=self.scale)

            if self.phase:
                figure_phase = get_figure(figure_name="Phase spectrum - Pol1")
                plt.phase_spectrum(x=in1, Fs=self.fs, Fc=self.fc)
                figure_phase = get_figure(figure_name="Phase spectrum - Pol2")
                plt.phase_spectrum(x=in2, Fs=self.fs, Fc=self.fc)
        



class PDF_Analyser(Analyser):
    """Display the PDF of a signal"""
    def __init__(self, exp=True, Tx=True, dual_mode=False, name="PDF"):
        self.exp = exp
        self.Tx = Tx
        self.dual_mode = dual_mode
        self.name = name

    def analyse(self, input_data):
        sns.set_theme(style="darkgrid")
        if self.dual_mode == False:
            if self.exp:
                if self.Tx:
                    figure_real = get_figure(figure_name="Experimental TX Real signal")
                    sns.histplot(np.real(input_data),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Tx Imaginary signal")
                    sns.histplot(np.imag(input_data),bins=50)
                else:
                    figure_real = get_figure(figure_name="Experimental RX Real signal")
                    sns.histplot(np.real(input_data),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Rx Imaginary signal")
                    sns.histplot(np.imag(input_data),bins=50)
            else:
                if self.Tx:
                    figure_real = get_figure(figure_name="Simulated TX Real signal")
                    sns.histplot(np.real(input_data),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Tx Imaginary signal")
                    sns.histplot(np.imag(input_data),bins=50)
                else:
                    figure_real = get_figure(figure_name="Simulated RX Real signal")
                    sns.histplot(np.real(input_data),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Rx Imaginary signal")
                    sns.histplot(np.imag(input_data),bins=50)
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            if self.exp:
                if self.Tx:
                    figure_real = get_figure(figure_name="Experimental TX Real signal - Pol1")
                    sns.histplot(np.real(in1),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Tx Imaginary signal - Pol1")
                    sns.histplot(np.imag(in1),bins=50)

                    figure_real = get_figure(figure_name="Experimental TX Real signal - Pol2")
                    sns.histplot(np.real(in2),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Tx Imaginary signal - Pol2")
                    sns.histplot(np.imag(in2),bins=50)
                else:
                    figure_real = get_figure(figure_name="Experimental RX Real signal - Pol1")
                    sns.histplot(np.real(in1),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Rx Imaginary signal - Pol1")
                    sns.histplot(np.imag(in1),bins=50)

                    figure_real = get_figure(figure_name="Experimental RX Real signal - Pol2")
                    sns.histplot(np.real(in2),bins=50)
                    figure_imag = get_figure(figure_name="Experimental Rx Imaginary signal - Pol2")
                    sns.histplot(np.imag(in2),bins=50)
            else:
                if self.Tx:
                    figure_real = get_figure(figure_name="Simulated TX Real signal - Pol1")
                    sns.histplot(np.real(in1),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Tx Imaginary signal - Pol1")
                    sns.histplot(np.imag(in1),bins=50)

                    figure_real = get_figure(figure_name="Simulated TX Real signal - Pol2")
                    sns.histplot(np.real(in2),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Tx Imaginary signal - Pol2")
                    sns.histplot(np.imag(in2),bins=50)
                else:
                    figure_real = get_figure(figure_name="Simulated RX Real signal - Pol1")
                    sns.histplot(np.real(in1),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Rx Imaginary signal - Pol1")
                    sns.histplot(np.imag(in1),bins=50)
                    figure_real = get_figure(figure_name="Simulated RX Real signal - Pol2")
                    sns.histplot(np.real(in2),bins=50)
                    figure_imag = get_figure(figure_name="Simulated Rx Imaginary signal - Pol2")
                    sns.histplot(np.imag(in2),bins=50)


class Density_2d(Analyser):
    """Display the 2D density of a signal"""
    def __init__(self, exp=True, Tx=True, dual_mode=False, name="2D-Density"):
        self.exp = exp
        self.Tx = Tx
        self.dual_mode = dual_mode
        self.name = name

    def get_bw_adj(self,x):
        if len(x)>100:
            x = x[:100]
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                {'bandwidth': bandwidths},
                cv=KFold(n_splits=2, shuffle=True))
        grid.fit(np.real(x[:, None]))
        bw_real = grid.best_params_["bandwidth"]
        grid.fit(np.imag(x[:, None]))
        bw_imag = grid.best_params_["bandwidth"]
        bw = np.mean(np.array([bw_real, bw_imag]))
        return bw
        

    def analyse(self, input_data):
        bw_fact = self.get_bw_adj(input_data)
        sns.set_theme(style="darkgrid")
        if self.dual_mode==False:
            if self.exp:
                if self.Tx:
                    plt.figure()
                    plt.title("Experimental TX")
                    sns.kdeplot(x=np.real(input_data), y=np.imag(input_data), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                else:
                    plt.figure()
                    plt.title("Experimental RX")
                    sns.kdeplot(x=np.real(input_data), y=np.imag(input_data), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
            else:
                if self.Tx:
                    plt.figure()
                    plt.title("Simulated TX")
                    sns.kdeplot(x=np.real(input_data), y=np.imag(input_data), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                else:
                    plt.figure()
                    plt.title("Simulated RX")
                    sns.kdeplot(x=np.real(input_data), y=np.imag(input_data), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
            plt.xlabel("Real data")
            plt.ylabel("Imaginary data")
            plt.axis("scaled")
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            if self.exp:
                if self.Tx:
                    plt.figure()
                    plt.title("Experimental TX - Pol1")
                    sns.kdeplot(x=np.real(in1), y=np.imag(in1), cmap="Blues", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                    plt.figure()
                    plt.title("Experimental TX - Pol2")
                    sns.kdeplot(x=np.real(in2), y=np.imag(in2), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                else:
                    plt.figure()
                    plt.title("Experimental RX - Pol1")
                    sns.kdeplot(x=np.real(in1), y=np.imag(in1), cmap="Blues", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                    plt.figure()
                    plt.title("Experimental RX - Pol2")
                    sns.kdeplot(x=np.real(in2), y=np.imag(in2), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
            else:
                if self.Tx:
                    plt.figure()
                    plt.title("Simulated TX - Pol1")
                    sns.kdeplot(x=np.real(in1), y=np.imag(in1), cmap="Blues", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                    plt.figure()
                    plt.title("Simulated TX - Pol2")
                    sns.kdeplot(x=np.real(in2), y=np.imag(in2), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                else:
                    plt.figure()
                    plt.title("Simulated RX - Pol1")
                    sns.kdeplot(x=np.real(in1), y=np.imag(in1), cmap="Blues", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
                    plt.figure()
                    plt.title("Simulated RX - Pol2")
                    sns.kdeplot(x=np.real(in2), y=np.imag(in2), cmap="Reds", shade=True, thresh=0.05, bw_method="scott", bw_adjust=bw_fact)
                    plt.xlabel("Real data")
                    plt.ylabel("Imaginary data")
                    plt.axis("scaled")
