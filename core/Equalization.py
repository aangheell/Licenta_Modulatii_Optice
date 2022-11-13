import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from core.Processors import Processor,Chain,Recorder
from numpy import dtype, linalg as LA
import scipy

class CD_compensation(Processor):
    '''
    Frequency domain equalizer of CD
    Parameters
    ----------
    L : float
        fiber lenght
    
    D: float
        Dispersion coefficient (17 ps/nm/km)

    Lambda: float
        Wavelength of the communication

    fs: float
        Sampling frequency

    Ts: float
        Symbol period

    name: str
        The name of the object

    Returns
    -------
    Compensated signal
    '''
    c = 3e8

    def __init__(self, L, D, Lambda=1550e-9, fs=32e9, name="CD compensation"):
        self.L = L
        self.D = D
        self.Lambda = Lambda
        self.fs = fs
        self.name = name

    def H(self,N=1024):
        f = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2,N, endpoint=False))
        w = 2*np.pi*f
        self.H_f = np.exp(1j* ( (self.L*self.D)* self.Lambda**2)/(4*np.pi*CD_compensation.c) *w**2 )
        return self.H_f

    def process(self,input_data):
        # get channel response
        N = len(input_data)
        H = self.H(N)
        # Frequency domain
        fft_input = np.fft.fft(input_data)
        fft_output = H * fft_input

        # Time domain
        output_data = np.fft.ifft(fft_output)
        return output_data


class Preamble_based_cfo(Processor):
    """
    Frequency offset estimation for preamble-based DSP. Uses a transmitted preamble
    sequence to find the frequency offset from the corresponding aligned symbols.
    
    Gives higher accuracy than blind power of 4 based FFT for noisy signals. 
    Calculates the phase variations between the batches and does a linear fit
    to find the corresponding frequency offset. 
    
    Parameters:
        Tx_symb: Complex transmitted symbols from which we extract the preamble
        
        preamble: the length of the preamble

        name: name of the object


    Output:
        Compensated signal
    
    """
    def __init__(self, Tx_symb, preamble, Ts, fs, name='Pilot_based_cfo' ):
        self.Tx_symb = Tx_symb
        self.preamble = preamble
        self.Ts = Ts
        self.fs = fs
        self.name = name
    

    def process(self,input_data):
        # Carrier frequency offset compensation
        N = len(input_data)
        Tx_preamble = getattr(self.Tx_symb, "data")[:self.preamble]
        Rx_preamble = input_data[:self.preamble]
        phaseEvolution = np.unwrap(np.angle(np.conj(Tx_preamble)*Rx_preamble))
        freqFit = np.polyfit(np.arange(0,len(phaseEvolution)),phaseEvolution,1)
        foe = freqFit[0]
        output_data = input_data*np.exp(-1j*np.arange(N)*foe)
        return output_data[self.preamble:]


class TA_CFO(Processor):
    '''
    Training-aided frequency estimation
    
    X. Zhou, X. Chen, and K. Long, “Wide-range frequency offset estimation algorithm for optical coherent systems using training sequence,” IEEE Photon. Technol. Lett., vol. 24, no. 1, pp. 82–84, Jan. 2012.
       
        Parameters
    ----------
    Tx_symb: object
        Used for extraction of transmitted piltots
    
    preamble: int
        Used for training symbols extraction

    name : str 

    Returns
    -------
    Compensated signal
    
    '''

    def __init__(self, Tx_symb, preamble, Ts, fs, name='Preamble_based_cfo'):
        self.Tx_symb = Tx_symb
        self.preamble = preamble
        self.Ts = Ts
        self.fs = fs
        self.name = name
    

    def process(self,input_data):
        N = len(input_data)
        Tx_preamble = getattr(self.Tx_symb, "data")[:self.preamble]
        Rx_preamble = input_data[:self.preamble]
        x_prime = Rx_preamble*np.conj(Tx_preamble)
        sum = 0
        for i in range(1,self.preamble):
            sum += x_prime[i] * np.conj(x_prime[i-1])
        Delta_f = np.angle(sum)
        output_data = input_data * np.exp(-1j*np.arange(N)*Delta_f)
        return output_data[self.preamble:]

        
class Blind_CFO(Processor):
    '''
    Leven, Andreas, et al. "Frequency estimation in intradyne reception." IEEE Photonics Technology Letters 19.6 (2007): 366-368

    Returns: 
    _ _ _ _
    Compensated signal
    '''

    def __init__(self, name='Blind_CFO'):
        self.name = name

    def process(self, input_data):
        N = len(input_data)
        sum = 0
        for i in range(1,N):
            sum += (input_data[i] * np.conj(input_data[i-1]))**4
        w0 = np.angle(sum)/4
        print(w0)
        output_data = input_data * np.exp(-1j*np.arange(N)*w0)
        return output_data

class Blind_CFO_PD(Processor):
    """
    Blind_CFO()
    Blind CFO estimator based on the maximisatio of the periodogram of 4th order statistic

    .. math::

        \widehat{\omega} = \frac{1}{4} \arg \max_{\omega} \|\sum_{n=0}^{N-1}x^4[n]e^{-j\omega n}\|^2

    The maximisation is performed using the Newton Algorithm

    Parameters
    ----------
    w0 : float
        Initialisation in rad/samples
    N_iter : int
        Number of iterations
    method : str 


    Returns
    -------
    Compensated signal

    Example
    -------

    .. code-block:: Python

 
    """

    def __init__(self,w0=0,N_iter=5, method = "newton",step_size=10**(-5),name = "cfo"):
        self.name = name
        self.w0 = w0
        self.N_iter = N_iter
        self.method = method
        self.step_size = step_size

    def cost(self,x,w):
        N = len(x)
        x4 = x**4
        dtft = self.compute_dtft(x4,w)
        return (np.abs(dtft)**2)/N

    def compute_dtft(self,x,w):
        N = len(x)
        N_vect = np.arange(N)
        dtft = np.sum(x*np.exp(-1j*w*N_vect))
        return dtft

    def fit(self,x,w0):
        w = w0
        N = len(x)
        x4 = x**4
        N_vect = np.arange(N)
        step_size = self.step_size

        for n in range(self.N_iter):

            #print("w0={} cost={}".format(w,self.cost(x,w)))

            if self.method == "grad":
                dtft = self.compute_dtft(x4,w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4,w)
                grad = (1/N)*(dtft_diff*np.conj(dtft)+dtft*np.conj(dtft_diff))
                h = step_size*grad 

            if self.method == "newton":
                dtft = self.compute_dtft(x4,w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4,w)
                dtft_diff2 = self.compute_dtft(-(N_vect**2)*x4,w)
                grad = (1/N)*(dtft_diff*np.conj(dtft)+dtft*np.conj(dtft_diff))
                J = (2/N)*(np.real(dtft_diff2*np.conj(dtft))+(np.abs(dtft_diff)**2))
                h = -grad/J
                
            w = w + h
            
        w0 = np.real(w)/4
        return w0

    def process(self,input_data):

        N = len(input_data)
        N_vect = np.arange(N)

        w0 = self.fit(input_data,4*self.w0)
        Fs = 30e9
        w0_vect = np.linspace(-2*np.pi*2.2e9/Fs,-2*np.pi*1.8e9/Fs,1000)
        cost = []
        for l in w0_vect:
            cost_temp = self.cost(input_data, 4*l)
            cost.append(cost_temp)
        
        plt.figure()
        plt.title('CFO Cost Function')
        print(w0_vect[np.argmax(np.array(cost))])
        plt.plot(w0_vect,np.array(cost))
        self.w0 = w0
        print(self.w0)
        x = input_data*np.exp(-1j*w0*N_vect)
        return x      


class Blind_IQ(Processor):
    """
    Blind_IQ()
    Blind IQ estimator based on the diagonalisation of the augmented covariance matrix. This compensation assumes the circularity of compensated signal


    Parameters
    ----------
    None


    Returns
    -------
    Compensated signal

    Example
    -------

    .. code-block:: Python
 
    """

    def __init__(self,name="iq_compensator"):
        self.name = name

    def process(self,input_data):
        x = input_data

        N = len(x)
        X = np.vstack([np.real(x),np.imag(x)])

        # compute covariance matrix
        R = (1/N)*np.matmul(X,np.transpose(X))
        # perform eigenvalue decomposition
        V,U = LA.eig(R)

        # perform whitening
        D = np.diag(1/np.sqrt(V))
        M  = np.matmul(D,np.transpose(U))
        Y = np.matmul(M,X)
        x = Y[0,:]+1j*Y[1,:]
        
        sigma_s = np.mean(np.abs(x)**2)
        gain = 1 / np.sqrt(sigma_s)
        output_data = x*gain
        return output_data

class GSOP_IQ(Processor):
    """
    BlindIQ estimator based on the GSOP.

    I. Fatadin, S. J. Savory, and D. Ives, “Compensation of quadrature imbalance in an optical QPSK coherent receiver,” IEEE Photon. Technol. Lett., vol. 20, no. 20, pp. 1733–1735, Oct. 2008.

    Parameters
    ----------
    None


    Returns
    -------
    Compensated signal
 
    """
    def __init__(self, name="IQ_imbalance compensation"):
        self.name=name

    def process(self, input_data):
        rI = np.real(input_data)
        rQ = np.imag(input_data)
        PI = np.mean(rI**2);
        #PQ = np.mean(rQ**2);
    
        Io = (1/np.sqrt(PI))*rI;
        rho = np.mean(rI*rQ);
        Qprime = rQ - rho*rI/PI;
        PQprime = np.mean(Qprime**2);
        Qo = Qprime/np.sqrt(PQprime);
    
        output_data = Io + 1j*Qo;
        
        # Normalization
        self.sigma_s = np.mean(np.abs(output_data)**2)
        self.gain = 1 / np.sqrt(self.sigma_s)
        output_data *= self.gain
        return output_data


class Pilot_based_cpe(Processor):
    'Correlation of pilots and averaging'
    def __init__(self, Tx_symb, config, preamble, avg, avg_ml, tech='CME', ML=True, rot=False, name='pilot_based_cpe'):
        self.Tx_symb = Tx_symb
        self.config = config
        self.preamble = preamble
        self.avg = avg
        self.avg_ml = avg_ml
        self.tech = tech
        self.ML = ML
        self.rot = rot
        self.name = name

    def get_pilots(self, sig):
        N = len(sig)
        s_vect = np.reshape(sig, (N,1), order='F')
        s_vect_ext = np.vstack((np.real(s_vect), np.imag(s_vect)))
        G = np.zeros((N//self.config,N))
        data_info = np.ones(N, dtype=bool)
        for i in range(N//self.config):
            G[i,(i+1)*self.config//2+i*self.config//2] = 1
            data_info[(i+1)*self.config//2+i*self.config//2] = False

        G_pilots_1 = np.hstack((G,np.zeros((N//self.config,N))))
        G_pilots_2 = np.hstack((np.zeros((N//self.config,N)),G))
        G_pilots = np.vstack((G_pilots_1, G_pilots_2))
        s_pilots = np.ravel(np.matmul(G_pilots,s_vect_ext))
        Np = len(s_pilots)//2
        s_pil_comp = s_pilots[:Np] + 1j*s_pilots[Np:]
        data_info = np.ravel(np.reshape(data_info,(N,1), order='F'))
        output_metric = np.ravel(s_vect[data_info])
        return (s_pil_comp, output_metric, data_info)

    def avg_filter(self, rx_pil, tx_pil, avg):
        N = len(rx_pil)
        avg_down = (avg//2 - 1)
        avg_up = avg//2
        rx_pil_ext = np.hstack( (np.zeros(avg_down, dtype=complex), rx_pil, np.zeros(avg_up, dtype=complex) ) )
        tx_pil_ext = np.hstack( (np.zeros(avg_down, dtype=complex), tx_pil, np.zeros(avg_up, dtype=complex) ) )
        avg_pil = np.zeros(N, dtype=complex)
        for  i in range(N):
            avg_pil[i] =  np.sum( rx_pil_ext[i:i+avg] * np.conj(tx_pil_ext[i:i+avg]) )
        return avg_pil

    def avg_chalmers(self, phase):
        ret = np.cumsum(np.insert(phase, 0,0))
        ret = (ret[self.avg:] - ret[:-self.avg])/self.avg
        return ret

    def process(self, input_data):
        N = len(input_data)
        Tx_symb = getattr(self.Tx_symb,'data')[self.preamble:]
        tx_pilot, output_tx, data_info_rx = self.get_pilots(Tx_symb)
        rx_pilot, output_rx, data_info_rx = self.get_pilots(input_data)
        if self.rot:
            ph_diff = np.mean(np.unwrap(np.angle(rx_pilot/tx_pilot)))
            output_data = input_data * np.exp(-1j*ph_diff)
            #rx_pilot, output_rx, data_info_rx = self.get_pilots(input_data)
        else:
            phase_pil = np.unwrap( np.angle( np.conj(tx_pilot)* rx_pilot) )
            if self.avg > 1:
                phase_pil = np.unwrap(np.angle(self.avg_filter(rx_pilot, tx_pilot, self.avg)))
            phase_est = np.zeros(N)
            if self.tech=='CME':
                for  i in range(N//self.config):
                    phase_est[i*self.config:(i+1)*self.config] = phase_pil[i]
            output_data = input_data * np.exp(-1j*phase_est)
            if self.ML:
                constellation = np.unique(Tx_symb)
                distances_P1 = np.abs(output_data.reshape(-1,1)-constellation)**2
                data_index_P1 = np.argmin(distances_P1,axis=1)
                data_proj = np.zeros_like(output_data)
                for k in range(len(data_index_P1)):
                    data_proj[k] = constellation[data_index_P1[k]]
                if self.avg_ml > 1:
                    phase_ml = self.avg_filter(output_data, data_proj, self.avg_ml)
                else:
                    phase_ml = np.conj(data_proj)*output_data
                phase_ml = np.arctan(np.imag(phase_ml)/np.real(phase_ml))
                output_data = output_data * np.exp(-1j*phase_ml)
        return output_data   


class PRDE_PMD(Processor):
    """
    Compensate for the PMD using a probabilistic radius directed equalizer (PRDE) based on 
    D. Lavery, M. Paskov, R. Maher, S. J. Savory and P. Bayvel, "Modified radius directed equaliser for high order QAM," 2015 European 
    Conference on Optical Communication (ECOC), 2015, pp. 1-3, doi: 10.1109/ECOC.2015.7341620.
    
    Parameters
    _ _ _ _ _ 

    M: float
       QAM constellation order

    N_taps: float
        Number of filter taps

    step: float 
        Update step size

    epochs: float
        Number of epochs

    Return
    Dual-polarized compensated signal
    """

    def __init__(self, M, os_fact, N_taps, step, epochs, normalized=True, name="PRDE"):
        self.M = M
        self.os = os_fact
        self.N_taps = N_taps
        self.step = step
        self.epochs = epochs
        self.type = "{}QAM".format(self.M)
        self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
        self.df = pd.read_csv(self.filename) 
        self.normalized = normalized
        self.name = name 


    def constellation(self):
        symbol = []
        for i in self.df.columns:
            symbol.append(np.complex(i.replace("i","j")))
        self.symbol = np.array(symbol)
        if self.normalized == True:
            self.sigma_s = np.mean(np.abs(self.symbol)**2)
            self.gain = 1 / np.sqrt(self.sigma_s)
            self.symbol = self.symbol * self.gain
        return self.symbol


    def radius(self, symbols):
        radii = []
        for i in range(len(symbols)):
            radius = np.sqrt( np.real(symbols[i])**2 + np.imag(symbols[i])**2 )
            radii.append(radius)

        radii = np.array(radii[:])
        uniq_radii = np.unique(radii)
        prob_radii = np.zeros(len(uniq_radii))
        for  l in range(len(uniq_radii)):
            prob_radii[l] = int(np.sum(np.equal(uniq_radii[l].item(), radii)))/self.M
        return [uniq_radii, prob_radii]


    def get_rad_prob(self):
        data_points = np.arange(self.M)
        symb = []
        constellation = self.constellation()
        for i in data_points:
            symb.append(constellation[i])
        const_points = np.array(symb)
        [radii, prob] = self.radius(const_points)
        return [radii, prob, const_points]

    def process(self, input_data):
        N = len(input_data)//2
        xin1 = input_data[:N]
        xin2 = input_data[N:]
        [radii, prob, constellation] = self.get_rad_prob()
        w11 = np.zeros(self.N_taps, dtype=complex)
        w11[0] = 1 
        w22 = np.zeros(self.N_taps, dtype=complex)
        w22[0] = 1
        w12 = np.zeros(self.N_taps, dtype=complex)
        w21 = np.zeros(self.N_taps, dtype=complex)
        error1 = np.zeros(N)
        error2 = np.zeros(N)
        rad_out1 = np.zeros(N)
        rad_out2 = np.zeros(N)
        for iteration in range(self.epochs):
            for i in range(0, N-self.N_taps):
                xn1 = xin1[i+self.N_taps:i:-1]
                xn2 = xin2[i+self.N_taps:i:-1]
                xout1 = np.dot(xn1 , w11) + np.dot(xn2, w12)
                xout2 = np.dot(xn2 , w22) + np.dot(xn1, w21)
                rad_out1 = np.abs(xout1)
                rad_out2 = np.abs(xout2)
                index_rad1 = np.argmin( np.abs(radii - rad_out1) )
                index_rad2 = np.argmin( np.abs(radii - rad_out2) )
                error1 = prob[index_rad1] * ( radii[index_rad1]**2 - rad_out1**2 )
                error2 = prob[index_rad2] * ( radii[index_rad2]**2 - rad_out2**2 )
                if i % self.os !=0:
                    w11 = w11 + self.step*error1*xout1*np.conj(xn1)
                    w12 = w12 + self.step*error1*xout1*np.conj(xn2)
                    w22 = w22 + self.step*error2*xout2*np.conj(xn2)
                    w21 = w21 + self.step*error2*xout2*np.conj(xn1)

        for iteration in range(self.epochs):
            for i in range(0, N-self.N_taps):
                xn1 = xin1[i+self.N_taps:i:-1]
                xn2 = xin2[i+self.N_taps:i:-1]
                xout1 = np.dot(xn1, w11) + np.dot(xn2, w12)
                xout2 = np.dot(xn2, w22) + np.dot(xn1, w21)
                index1 = np.argmin(np.abs(xout1-constellation)**2)
                index2 = np.argmin(np.abs(xout2-constellation)**2)
                error1 = constellation[index1] - xout1
                error2 = constellation[index2] - xout2
                if i % self.os != 0:
                    w11 = w11 + self.step*error1*np.conj(xn1)
                    w12 = w12 + self.step*error1*np.conj(xn2)
                    w22 = w22 + self.step*error2*np.conj(xn2)
                    w21 = w21 + self.step*error2*np.conj(xn1)

        xout1 = np.convolve(xin1,w11)[:N] + np.convolve(xin2,w12)[:N]
        xout2 = np.convolve(xin2,w22)[:N] + np.convolve(xin1,w21)[:N]
        output_data = np.ravel(np.hstack( (xout1, xout2) ))
        return output_data



class CMA_PMD(Processor):
    """
    Faruk, Md Saifuddin, and Seb J. Savory. "Digital signal processing for coherent transceivers employing multilevel formats." 
    Journal of Lightwave Technology 35.5 (2017): 1125-1141.
    
    Parameters
    _ _ _ _ _ 

    M: float
       QAM constellation order

    N_taps: float
        Number of filter taps

    step: float 
        Update step size

    epochs: float
        Number of epochs

    Return
    Dual-polarized compensated signal
    """

    def __init__(self, M,  os_fact, N_taps, step, epochs, normalized=True, name="PRDE"):
        self.M = M
        self.os = os_fact
        self.N_taps = N_taps
        self.step = step
        self.epochs = epochs
        self.type = "{}QAM".format(self.M)
        self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
        self.df = pd.read_csv(self.filename) 
        self.normalized = normalized
        self.name = name 


    def constellation(self):
        symbol = []
        for i in self.df.columns:
            symbol.append(np.complex(i.replace("i","j")))
        self.symbol = np.array(symbol)
        if self.normalized == True:
            self.sigma_s = np.mean(np.abs(self.symbol)**2)
            self.gain = 1 / np.sqrt(self.sigma_s)
            self.symbol = self.symbol * self.gain
        return self.symbol


    def get_rad_prob(self):
        data_points = np.arange(self.M)
        symb = []
        constellation = self.constellation()
        for i in data_points:
            symb.append(constellation[i])
        const_points = np.array(symb)
        radius  = np.mean(np.abs(const_points)**4)/np.mean(np.abs(const_points)**2)
        return radius, const_points

    def process(self, input_data):
        N = len(input_data)//2
        xin1 = input_data[:N]
        xin2 = input_data[N:]
        radius, constellation = self.get_rad_prob()
        w11 = np.zeros(self.N_taps, dtype=complex)
        w11[0] = 1 
        w22 =  np.zeros(self.N_taps, dtype=complex) 
        w22[0] = 1
        w12 = np.zeros(self.N_taps, dtype=complex) 
        w21 = np.zeros(self.N_taps, dtype=complex) 
        for iteration in range(self.epochs-self.epochs//2):
            for i in range(0, N-self.N_taps):
                xn1 = xin1[i+self.N_taps:i:-1]
                xn2 = xin2[i+self.N_taps:i:-1]
                xout1 = np.dot(xn1, w11) + np.dot(xn2,w12)
                xout2 = np.dot(xn2, w22) + np.dot(xn1, w21)
                error1 = radius - np.abs(xout1)**2
                error2 = radius - np.abs(xout2)**2
                if i % self.os != 0:
                    w11 = w11 + self.step*error1*xout1*np.conj(xn1)
                    w12 = w12 + self.step*error1*xout1*np.conj(xn2)
                    w22 = w22 + self.step*error2*xout2*np.conj(xn2)
                    w21 = w21 + self.step*error2*xout2*np.conj(xn1)
        
        for iteration in range(self.epochs+self.epochs//2):
            for i in range(0, N-self.N_taps):
                xn1 = xin1[i+self.N_taps:i:-1]
                xn2 = xin2[i+self.N_taps:i:-1]
                xout1 = np.dot(xn1, w11) + np.dot(xn2,w12)
                xout2 = np.dot(xn2, w22) + np.dot(xn1, w21)
                index1 = np.argmin(np.abs(xout1-constellation)**2)
                index2 = np.argmin(np.abs(xout2-constellation)**2)
                error1 = constellation[index1] - xout1
                error2 = constellation[index2] - xout2
                if i % self.os != 0:
                    w11 = w11 + self.step*error1*np.conj(xn1)
                    w12 = w12 + self.step*error1*np.conj(xn2)
                    w22 = w22 + self.step*error2*np.conj(xn2)
                    w21 = w21 + self.step*error2*np.conj(xn1)
        
        xout1 = np.convolve(xin1,w11)[:N] + np.convolve(xin2,w12)[:N]
        xout2 = np.convolve(xin2,w22)[:N] + np.convolve(xin1,w21)[:N]
        output_data = np.ravel(np.hstack( (xout1, xout2) ))
        return output_data

class IQ_TX_Compensation(Processor):
    '''Fludger, Chris RS, and Theo Kupfer. "Transmitter impairment mitigation and monitoring for high baud-rate, high order modulation systems." 
    ECOC 2016; 42nd European Conference on Optical Communication. VDE, 2016.'''
    def __init__(self, Tx_symb, config, preamble, N_taps, epochs, step, extract=False, name='iq_tx_comp'):
        self.Tx_symb = Tx_symb
        self.config = config
        self.preamble = preamble
        self.N_taps = N_taps
        self.epochs = epochs
        self.step = step
        self.extract = extract
        self.name = name

    def get_pilots(self, sig):
        N = len(sig)
        s_vect = np.reshape(sig, (N,1), order='F')
        s_vect_ext = np.vstack((np.real(s_vect), np.imag(s_vect)))
        G = np.zeros((N//self.config,N))
        data_info = np.ones(N, dtype=bool)
        for i in range(N//self.config):
            G[i,(i+1)*self.config//2+i*self.config//2] = 1
            data_info[(i+1)*self.config//2+i*self.config//2] = False

        G_pilots_1 = np.hstack((G,np.zeros((N//self.config,N))))
        G_pilots_2 = np.hstack((np.zeros((N//self.config,N)),G))
        G_pilots = np.vstack((G_pilots_1, G_pilots_2))
        s_pilots = np.ravel(np.matmul(G_pilots,s_vect_ext))
        Np = len(s_pilots)//2
        s_pil_comp = s_pilots[:Np] + 1j*s_pilots[Np:]
        data_info = np.ravel(np.reshape(data_info,(N,1), order='F'))
        output_metric = np.ravel(s_vect[data_info])
        return (s_pil_comp, output_metric, data_info)


    def process(self, input_data):
        Tx_symb = getattr(self.Tx_symb,'data')[self.preamble:]
        tx_pilot, output_tx, data_info_rx = self.get_pilots(Tx_symb)
        rx_pilot, output_rx, data_info_rx = self.get_pilots(input_data)
        rx_pilot = np.hstack( (rx_pilot, np.zeros(self.N_taps, dtype=complex)) )
        tx_pilot = np.hstack( (tx_pilot, np.zeros(self.N_taps, dtype=complex)) )
        RXI_pilot = np.real(rx_pilot)
        RXQ_pilot = np.imag(rx_pilot)
        TXI_pilot = np.real(tx_pilot)
        TXQ_pilot = np.imag(tx_pilot)
        N_pil = len(RXI_pilot)
        hii = np.zeros(self.N_taps)
        hii[0] = 1 
        hqq =  np.zeros(self.N_taps) 
        hqq[0] = 1
        hiq = np.zeros(self.N_taps) 
        hqi = np.zeros(self.N_taps)
        self.ERR1 = []
        self.ERR2 = []
        for iteration in range(self.epochs):
            for i in range(0, N_pil-self.N_taps):
                tempI = RXI_pilot[i:i+self.N_taps]
                tempQ = RXQ_pilot[i:i+self.N_taps]
                mi = np.dot(tempI, hii) + np.dot(tempQ,hqi)
                mq = np.dot(tempQ, hqq) + np.dot(tempI, hiq)
                error1 = TXI_pilot[i] - mi
                error2 = TXQ_pilot[i] - mq
                self.ERR1.append(error1)
                self.ERR2.append(error2)
                hii = hii + self.step*error1*tempI
                hqi = hqi + self.step*error1*tempQ
                hiq = hiq + self.step*error2*tempI
                hqq = hqq + self.step*error2*tempQ
     
        if self.extract:
            real_output = np.real(output_rx)
            imag_output = np.imag(output_rx)
            N = len(output_rx)
            output_data = (np.convolve(real_output, hii)[:N]+np.convolve(imag_output, hqi)[:N]) + 1j*(np.convolve(real_output, hiq)[:N]+np.convolve(imag_output, hqq)[:N])
        else:
            real_output = np.real(input_data)
            imag_output = np.imag(input_data)
            N = len(input_data)
            output_data = (np.convolve(real_output, hii)[:N]+np.convolve(imag_output, hqi)[:N]) + 1j*(np.convolve(real_output, hiq)[:N]+np.convolve(imag_output, hqq)[:N])
        return output_data   


class Constant_phase_rotation(Processor):
    def __init__(self, tx_symb, Np, name='Phase_rotation'):
        self.tx_symb = tx_symb
        self.Np = Np
        self.name = name

    def process(self,input_data):
        tx_symb = getattr(self.tx_symb, 'data')[self.Np:]
        rx_symb = input_data
        ph_diff = np.mean(np.unwrap(np.angle(rx_symb/tx_symb)))
        output_data = input_data * np.exp(-1j*ph_diff)
        return output_data
