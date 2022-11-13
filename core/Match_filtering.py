import numpy as np
from scipy.signal import upfirdn
import matplotlib.pyplot as plt

from core.Processors import Processor

class RRC_Resample(Processor):
    def __init__(self, N, alpha, Ts, Fs, os, dual_mode=False, name='RRC Filter', **kwargs):
        self.N = N
        self.alpha = alpha
        self.Ts = Ts
        self.Fs = Fs
        self.os = os
        self.dual_mode = dual_mode
        self.name = name
        if kwargs:
            for key, value in kwargs.items():
                if key=='pos':
                    self.config = value
        else:
            self.config = 'Tx'


    def get_filter_h(self):
        T_delta = 1/float(self.Fs)
        time_idx = ((np.arange(self.N)-self.N/2))*T_delta
        sample_num = np.arange(self.N)
        h_rrc = np.zeros(self.N, dtype=float)
            
        for x in sample_num:
            t = (x-self.N/2)*T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - self.alpha + (4*self.alpha/np.pi)
            elif self.alpha != 0 and t == self.Ts/(4*self.alpha):
                h_rrc[x] = (self.alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*self.alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*self.alpha)))))
            elif self.alpha != 0 and t == -self.Ts/(4*self.alpha):
                h_rrc[x] = (self.alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*self.alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*self.alpha)))))
            else:
                h_rrc[x] = (np.sin(np.pi*t*(1-self.alpha)/self.Ts) +  \
                        4*self.alpha*(t/self.Ts)*np.cos(np.pi*t*(1+self.alpha)/self.Ts))/ \
                        (np.pi*t*(1-(4*self.alpha*t/self.Ts)*(4*self.alpha*t/self.Ts))/self.Ts)

        # Scale #
        h_rrc = np.hstack((h_rrc,h_rrc[0]))
        h_rrc = h_rrc / np.sqrt(np.sum(h_rrc**2))
        return h_rrc

    def process(self, input_data):
        span = self.N // self.os
        h_rrc = self.get_filter_h()
        if self.dual_mode == False:
            real_data = np.real(input_data)
            imag_data = np.imag(input_data)
            if self.config == "Tx":
                output_real = upfirdn(h_rrc, real_data, up=self.os)
                output_imag = upfirdn(h_rrc, imag_data, up=self.os)
                output_data = output_real + 1j*output_imag
                output_data  = output_data

            elif self.config == "Rx":
                output_real = upfirdn(h_rrc, real_data, down=self.os)
                output_imag = upfirdn(h_rrc, imag_data, down=self.os)
                output_data = output_real + 1j*output_imag
                output_data = output_data[span:-span]
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            real_data1 = np.real(in1)
            imag_data1 = np.imag(in1)
            real_data2 = np.real(in2)
            imag_data2 = np.imag(in2)
            if self.config == "Tx":
                out_real1 = upfirdn(h_rrc, real_data1, up=self.os)
                out_imag1 = upfirdn(h_rrc, imag_data1, up=self.os)
                out1 = out_real1 + 1j*out_imag1
                out_real2 = upfirdn(h_rrc, real_data2, up=self.os)
                out_imag2 = upfirdn(h_rrc, imag_data2, up=self.os)
                out2 = out_real2 + 1j*out_imag2
                output_data  = np.ravel(np.hstack( (out1, out2 )))

            elif self.config == "Rx":
                out_real1 = upfirdn(h_rrc, real_data1, down=self.os)
                out_imag1 = upfirdn(h_rrc, imag_data1, down=self.os)
                out1 = out_real1 + 1j*out_imag1
                out_real2 = upfirdn(h_rrc, real_data2, down=self.os)
                out_imag2 = upfirdn(h_rrc, imag_data2, down=self.os)
                out2 = out_real2 + 1j*out_imag2
                output_data  = np.ravel(np.hstack( (out1[span:-span], out2[span:-span]) ))         
        return output_data

class RRC(Processor):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    def __init__(self, N, alpha, Ts, Fs, dual_mode=False, name='RRC Filter', **kwargs):
        self.N = N
        self.alpha = alpha
        self.Ts = Ts
        self.Fs = Fs
        self.dual_mode = dual_mode
        self.name = name
        if kwargs:
            for key, value in kwargs.items():
                if key=='pos':
                    self.config = value
        else:
            self.config = 'Tx'


    def get_filter_h(self):
        T_delta = 1/float(self.Fs)
        time_idx = ((np.arange(self.N)-self.N/2))*T_delta
        sample_num = np.arange(self.N)
        h_rrc = np.zeros(self.N, dtype=float)
        for x in sample_num:
            t = (x-self.N/2)*T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - self.alpha + (4*self.alpha/np.pi)
            elif self.alpha != 0 and t == self.Ts/(4*self.alpha):
                h_rrc[x] = (self.alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*self.alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*self.alpha)))))
            elif self.alpha != 0 and t == -self.Ts/(4*self.alpha):
                h_rrc[x] = (self.alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*self.alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*self.alpha)))))
            else:
                h_rrc[x] = (np.sin(np.pi*t*(1-self.alpha)/self.Ts) +  \
                        4*self.alpha*(t/self.Ts)*np.cos(np.pi*t*(1+self.alpha)/self.Ts))/ \
                        (np.pi*t*(1-(4*self.alpha*t/self.Ts)*(4*self.alpha*t/self.Ts))/self.Ts)

        # Scale #
        h_rrc = np.hstack((h_rrc,h_rrc[0]))
        h_rrc = h_rrc / np.sqrt(np.sum(h_rrc**2))
        #plt.figure()
        #plt.plot(np.arange(len(h_rrc)), h_rrc)
        #plt.show()
        return h_rrc

    def process(self, input_data):
        span = self.N
        h_rrc = self.get_filter_h()
        if self.dual_mode == False:
            real_data = np.real(input_data)
            imag_data = np.imag(input_data)
            if self.config == "Tx":
                output_real = np.convolve(h_rrc, real_data)
                output_imag = np.convolve(h_rrc, imag_data)
                output_data = output_real + 1j*output_imag
                output_data = output_data

            elif self.config == "Rx":
                output_real = np.convolve(h_rrc, real_data)
                output_imag = np.convolve(h_rrc, imag_data)
                output_data_delay = output_real + 1j*output_imag
                output_data = output_data_delay[span:-span]
        else:
            in1 = input_data[:len(input_data)//2]
            in2 = input_data[len(input_data)//2:]
            real_data1 = np.real(in1)
            imag_data1 = np.imag(in1)
            real_data2 = np.real(in2)
            imag_data2 = np.imag(in2)
            if self.config == "Tx":
                out_real1 = np.convolve(h_rrc, real_data1)
                out_imag1 = np.convolve(h_rrc, imag_data1)
                out1 = out_real1 + 1j*out_imag1
                out_real2 = np.convolve(h_rrc, real_data2)
                out_imag2 = np.convolve(h_rrc, imag_data2)
                out2 = out_real2 + 1j*out_imag2
                output_data = np.ravel( np.hstack( (out1,out2) ) )

            elif self.config == "Rx":
                out_real1 = np.convolve(h_rrc, real_data1)
                out_imag1 = np.convolve(h_rrc, imag_data1)
                out1 = out_real1 + 1j*out_imag1
                out_real2 = np.convolve(h_rrc, real_data2)
                out_imag2 = np.convolve(h_rrc, imag_data2)
                out2 = out_real2 + 1j*out_imag2
                output_data = np.ravel( np.hstack( (out1[span:-span],out2[span:-span]) ) )
        return output_data