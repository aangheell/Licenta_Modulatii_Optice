
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.type_check import iscomplex
from scipy.linalg import dft, sqrtm
from scipy.signal import resample
import pandas as pd
from sympy import arg
from core.Processors import Processor,Chain,Recorder

class Random_matrix(Processor):
    def __init__(self, name="processor"):
        self.name = name

    def process(self, input_data):
        H = pd.read_csv('./Top06_30_20.csv', header=None).to_numpy()
        if iscomplex(input_data).any():
            aug_data = np.hstack((np.real(input_data), np.imag(input_data)))
            output_data_aug = np.matmul(H, aug_data)
            output_data = output_data_aug[:,:input_data.shape[1]] + 1j*output_data_aug[:,input_data.shape[1]:] 
        else:
            output_data = np.matmul(H,input_data)
        return output_data


class FIR_channel(Processor):
    """Apply a FIR channel"""
    def __init__(self, h, name='FIR channel'):
        self.h = h
        self.name = name

    def process(self, input_data):
        N = len(input_data)
        L = len(self.h)
        real_data = np.real(input_data)
        imag_data = np.imag(input_data)
        real_out = np.convolve(real_data, np.real(self.h))[:N]
        imag_out = np.convolve(imag_data, np.imag(self.h))[:N]
        output_data = real_out + 1j*imag_out
        return output_data

class Fractional_delay(Processor):
    """Generate a fractional delay filter"""
    def __init__(self, N_filter, tau, name="FD Filter"):
        self.N_filter = N_filter
        self.tau = tau
        self.name = name

    def h_filter(self):
        # Compute the filter
        h_filt = np.sinc( np.arange(self.N_filter) - self.tau )
        # Multiply sinc filter by window
        h_filt *= np.hamming(self.N_filter)
        # Normalize to get unity gain.
        h_filt /= np.sum(h_filt)
        return h_filt

    def process(self, input_data):
        h = self.h_filter()
        output_data = np.convolve(input_data,h)[:input_data.shape[0]]
        return output_data

class LPN(Processor):
    '''Introduce Laser Phase Noise'''
    def __init__(self, df=0, Ts=1e-9, dual=False, pos='Tx', name='LPN'):
        self.df = df
        self.fs = 1/Ts
        self.dual = dual
        self.pos = pos
        self.name = name

    def phase_noise(self,sz):
        var = 2*np.pi*self.df/self.fs
        f = np.random.normal(scale=np.sqrt(var), size=sz)
        if len(f.shape) > 1:
            return np.cumsum(f, axis=1)
        else:
            return np.cumsum(f)

    def process(self,input_data):
        if self.dual == False:
            N = len(input_data)
            if  self.pos == 'Tx':
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                np.random.seed()
            elif self.pos == 'Rx':
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                np.random.seed()
            output_data = input_data * np.exp(1j * self.ph[-(N+300000):-300000])
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            if  self.pos == 'Tx':
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                np.random.seed()
            elif self.pos == 'Rx':
                np.random.seed(18)
                self.ph = self.phase_noise(int(2e6))
                np.random.seed()
            out1 = in1 * np.exp(1j * (self.ph[-(N+660000):-660000]))
            out2 = in2 * np.exp(1j * (self.ph[-(N+660000):-660000]))
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data


class LPN_Block(Processor):
    '''Introduce Laser Phase Noise'''
    def __init__(self, Nb_signal, iter, df=0, Ts=1e-9, dual=False, pos="Tx", name='LPN_block'):
        self.Nb_signal = Nb_signal*1000
        self.iter = iter
        self.df = df
        self.fs = 1/Ts
        self.pos = pos
        self.dual = dual
        self.name = name

    def phase_noise(self,sz):
        var = 2*np.pi*self.df/self.fs
        self.theta = np.random.normal(scale=np.sqrt(var))
        f = np.random.normal(scale=np.sqrt(var), size=sz)
        if len(f.shape) > 1:
            return np.cumsum(f, axis=1)
        else:
            return np.cumsum(f)

    def process(self,input_data):
        if self.dual == False:
            N_block = len(input_data)
            if self.pos == "Tx":
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                self.ph = self.ph[-self.Nb_signal*N_block:]
                np.random.seed()
            elif self.pos == "Rx":
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                self.ph = self.ph[-self.Nb_signal*N_block:]
                np.random.seed()
            output_data = input_data * np.exp(1j * (self.ph[int(self.iter*N_block):int((self.iter+1)*N_block)]))
        else:
            N_block = len(input_data)//2
            in1 = input_data[:N_block]
            in2 = input_data[N_block:]
            if self.pos == "Tx":
                np.random.seed(17)
                self.ph = self.phase_noise(int(2e6))
                self.ph = self.ph[-self.Nb_signal*N_block:]
                np.random.seed()
            elif self.pos == "Rx":
                np.random.seed(18)
                self.ph = self.phase_noise(int(2e6))
                self.ph = self.ph[-self.Nb_signal*N_block:]
                np.random.seed()
            out1 = in1 * np.exp(1j * (self.ph[int(self.iter*N_block):int((self.iter+1)*N_block)]))
            out2 = in2 * np.exp(1j * (self.ph[int(self.iter*N_block):int((self.iter+1)*N_block)]))
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data


class CFO(Processor):
    '''Introduce Carrier Frequency Offset'''
    def __init__(self, w_delta, dual_mode=False, name="Carrier_Frequency_Offset"):
        self.w_delta = w_delta
        self.dual_mode = dual_mode
        self.name = name

    def process(self,input_data):
        if self.dual_mode == False:
            N = len(input_data)
            offset = np.exp(1j*np.arange(10*N,11*N)*self.w_delta)
            output_data = offset*input_data
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            offset = np.exp(1j*np.arange(10*N,11*N)*self.w_delta)
            out1 = offset*in1
            out2 = offset*in2
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data

class IQ_Imbalance_albe(Processor):
    ''' Introduce some IQ Imbalance '''
    def __init__(self, alpha_db = 1, theta_deg = 2, *args, dual_mode=False, name="IQ_Imbalance"):
        self.alpha_db1 = alpha_db
        self.theta_deg1 = theta_deg
        if args:
            self.alpha_db2 = args[0]
            self.theta_deg2 = args[1] 
        self.dual_mode = dual_mode
        self.name = name
        self.alpha1 = 10 ** (self.alpha_db1/10) - 1
        self.theta_rad1 = np.radians(self.theta_deg1)
        if self.dual_mode == True:
            self.alpha2 = 10 ** (self.alpha_db2/10) - 1
            self.theta_rad2 = np.radians(self.theta_deg2)

    def process(self,input_data):
        if self.dual_mode == False:
            self.mu = np.cos(self.theta_rad1/2) + 1j*self.alpha1*np.sin(self.theta_rad1/2)
            self.nu = self.alpha1*np.cos(self.theta_rad1/2) - 1j*np.sin(self.theta_rad1/2)
            output_data = self.mu * input_data + self.nu * np.conj(input_data)
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            self.mu1 = np.cos(self.theta_rad1/2) + 1j*self.alpha1*np.sin(self.theta_rad1/2)
            self.nu1 = self.alpha1*np.cos(self.theta_rad1/2) - 1j*np.sin(self.theta_rad1/2)
            out1 = self.mu1 * in1 + self.nu1 * np.conj(in1)
            self.mu2 = np.cos(self.theta_rad2/2) + 1j*self.alpha2*np.sin(self.theta_rad2/2)
            self.nu2 = self.alpha2*np.cos(self.theta_rad2/2) - 1j*np.sin(self.theta_rad2/2)
            out2 = self.mu2 * in2 + self.nu2 * np.conj(in2)
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data

class IQ_Imbalance_munu(Processor):
    
    ''' Introduce some IQ Imbalance '''

    def __init__(self,mu,nu,*args, dual_mode=False, name="IQ_Imbalance"):
        self.mu1 = mu
        self.nu1 = nu
        self.mu2 = args[0]
        self.nu2 = args[1]
        self.dual_mode = dual_mode
        self.name = name

    def process(self,input_data):
        if self.dual_mode == False:
            output_data = self.mu1 * input_data + self.nu1 * np.conj(input_data)
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = self.mu1 * in1 + self.nu1 * np.conj(in1)
            out2 = self.mu2 * in2 + self.nu2 * np.conj(in2)
            output_data = np.ravel(np.hstack( (out1, out2) ))
        return output_data

class PMD(Processor):

    '''Insert PMD based on  Ip, Ezra, and Joseph M. Kahn. "Digital equalization of chromatic dispersion and polarization mode dispersion." Journal of Lightwave 
    Technology 25.8 (2007): 2033-2043.
    
    Parameters
    _ _ _ _ _ 

    theta: float
        angle of the principle axis to the observed axis

    t_dgd: float
        diffrential group delay between the polarization axes

    fs: float
        sampling frequency
    
    Returns
    _ _ _ _

    Dual polarisation signal with PMD
    '''

    def __init__(self, theta, t_dgd, fs, name="PMD"):
        self.theta = theta
        self.t_dgd = t_dgd
        self.fs = fs
        self.name = name
        
    def rot_mat(self,theta):
        H = np.array([ [np.cos(theta), np.sin(theta) ], [-np.sin(theta), np.cos(theta)] ])
        return H

    def process(self,input_data):
        N = len(input_data)
        w = 2*np.pi*np.linspace(-self.fs/2, self.fs/2, N//2, endpoint=False)
        x_in = np.vstack( (input_data[:N//2], input_data[N//2:]) )
        X_in_freq = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_in, axes=1),axis=1), axes=1)
        H1 = self.rot_mat(self.theta)
        X_rot1 = np.matmul(H1, X_in_freq)
        H2 = np.array([np.exp(1j*w*self.t_dgd/2), np.exp(-1j*w*self.t_dgd/2)])
        X_bir_freq = H2*X_rot1
        H3 = self.rot_mat(-self.theta)
        X_rot2 = np.matmul(H3, X_bir_freq)
        x_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_rot2, axes=1),axis=1), axes=1)
        output_data = np.ravel( np.reshape(x_out, (N,1), order='C') )
        return output_data    
