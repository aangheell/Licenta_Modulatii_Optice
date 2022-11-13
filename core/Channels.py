from core.Processors import Processor
import numpy as np
from scipy.stats import norm
from numpy.fft import fft,fftshift,ifft
import matplotlib.pyplot as plt

class AWGN(Processor):
    type = "channel"

    def __init__(self, SNR_db = 30, os = 1, is_complex = True, dual_mode = False, name = "AWGN"):
        self.SNR_db = SNR_db
        self.os = os
        self.is_complex = is_complex
        self.dual_mode = dual_mode
        self.name = name

    def get_noise(self, N, power):
        self.sigma2 = (power * 10 ** (-self.SNR_db/10)) * self.os
        if self.is_complex == True:
            sigma2_r = np.sqrt(self.sigma2/2)
            noise = norm.rvs(scale= sigma2_r, size=N)+1j*norm.rvs(scale=sigma2_r,size=N)
        else:
            sigma2_r = np.sqrt(self.sigma2)
            noise = norm.rvs(scale= sigma2_r, size= N)
        return noise
            
    def get_output_data(self,input_data):
        return input_data
 
    def process(self,input_data):
        if self.dual_mode == False:
            output_data = self.get_output_data(input_data)
            N = len(output_data)
            signal_power = np.mean(np.abs(output_data)**2)
            self.noise = self.get_noise(N,signal_power)
            output_data = output_data + self.noise
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = self.get_output_data(in1)
            out2 = self.get_output_data(in2)
            signal_power1 = np.mean(np.abs(out1)**2)
            self.noise1 = self.get_noise(N,signal_power1)
            out1 = out1 + self.noise1
            signal_power2 = np.mean(np.abs(out2)**2)
            self.noise2 = self.get_noise(N,signal_power2)
            out2 = out2 + self.noise2
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data

class Selective_AWGN(AWGN):
    
    def __init__(self,h=np.array([1]), SNR_db = 30,is_complex = True,fs=1, dual_mode=False, name="Selective Channel"):
        self.h = h
        self.fs = fs
        self.SNR_db = SNR_db
        self.dual_mode = dual_mode
        self.is_complex = is_complex
        self.name = name

    def get_output_data(self,input_data):
        # filter (convolve function seems to be the fastest)
        output = np.convolve(input_data,self.h)
        return output[:len(input_data)]

class CD_Channel(Processor):

    c = 3e8

    def __init__(self, gamma, Lambda = 1550e-9, fs=10e9, dual_mode=False, name="Chromatic Dispersion"):
        self.gamma = gamma                  # parameter describing chromatic dispersion
        self.Lambda = Lambda
        self.fs = fs
        self.dual_mode = dual_mode
        self.name = name

    def H(self,N=1024):
        f = fftshift(np.linspace(-self.fs/2,self.fs/2,N, endpoint=False))
        w = 2*np.pi*f
        self.H_f = np.exp(-1j* (self.gamma* self.Lambda**2)/(4*np.pi*CD_Channel.c) *w**2 )
        return self.H_f

    def process(self,input_data):
        if self.dual_mode == False:
            # get channel response
            N = len(input_data)
            H = self.H(N)
            # Frequency domain
            fft_input = fft(input_data)
            fft_output = H * fft_input

            # Time domain
            output_data = ifft(fft_output)
        else:
            # get channel response
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            H = self.H(N)
            # Frequency domain
            fft_input1 = fft(in1)
            fft_output1 = H * fft_input1
            out1 = ifft(fft_output1)
            fft_input2 = fft(in2)
            fft_output2 = H * fft_input2
            out2 = ifft(fft_output2)
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data


class CD_Channel_Block(CD_Channel):

    c = 3e8

    def __init__(self, BL, gamma, Lambda = 1550e-9, fs=10e9, dual_mode=False, name="Chromatic Dispersion"):
        self.BL = BL
        self.gamma = gamma                
        self.Lambda = Lambda
        self.fs = fs
        self.dual_mode = dual_mode
        self.name = name


    def process(self,input_data):
        if self.dual_mode == False:
            # get channel response
            N = len(input_data)
            H = self.H(self.BL)
            out = []
            # Frequency domain
            for i in range(N//self.BL):
                fft_input = fft(input_data[i*self.BL:(i+1)*self.BL])
                fft_output = H * fft_input

                # Time domain
                out.append(ifft(fft_output))
            output_data = np.ravel(out)
        else:
            # get channel response
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            H = self.H(self.BL)
            out1 = []
            out2 = []
            # Frequency domain
            for i in range(N//self.BL):
                fft_input1 = fft(in1[i*self.BL:(i+1)*self.BL])
                fft_output1 = H * fft_input1
                out1.append(ifft(fft_output1))
                fft_input2 = fft(in2[i*self.BL:(i+1)*self.BL])
                fft_output2 = H * fft_input2
                out2.append(ifft(fft_output2))
            out1 = np.ravel(out1)
            out2 = np.ravel(out2)
            output_data = np.ravel( np.hstack( (out1, out2) ) )
        return output_data

class Nonlinear_channel(Processor):
    c = 3e8

    def __init__(self, fs, l_f, n_span, n_seg, Dz, Lambda, theta, t_dgd, gamma, name='NLC'):
        self.fs = fs
        self.l_f = l_f
        self.n_span = n_span
        self.n_seg = n_seg
        self.gamma_nl = gamma
        self.Lambda = Lambda
        self.Dz = Dz
        self.theta = theta
        self.t_dgd = t_dgd
        self.name = name

    def H(self, N=1024):
        f = fftshift(np.linspace(-self.fs/2,self.fs/2,N, endpoint=False))
        w = 2*np.pi*f
        self.H_f = np.exp(-1j* (self.Dz* self.Lambda**2)/(4*np.pi*Nonlinear_channel.c) *w**2 )
        return self.H_f

    def rot_mat(self,theta):
        H = np.array([ [np.cos(theta), np.sin(theta) ], [-np.sin(theta), np.cos(theta)] ])
        return H

    def process(self, input_data):
        N = len(input_data)//2
        step_s = (self.l_f/((self.n_seg)*(self.n_span)))
        in1 = input_data[:N]
        in2 = input_data[N:]
        H = self.H(N)
        # Frequency domain
        for j in range(self.n_span):
            for jj in range(self.n_seg):
                fft_input1 = fft(in1)
                fft_input2 = fft(in2)
                fft_output1 = H * fft_input1
                fft_output2 = H * fft_input2
                out1 = ifft(fft_output1)
                out2 = ifft(fft_output2)
                input_temp = np.ravel( np.hstack( (out1, out2) ) )
                in_temp1 = input_temp[:N]
                in_temp2 = input_temp[N:]

                w = 2*np.pi*np.linspace(-self.fs/2, self.fs/2, N, endpoint=False)
                x_in = np.vstack( (in_temp1, in_temp2) )
                X_in_freq = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_in, axes=1),axis=1), axes=1)
                H1 = self.rot_mat(self.theta)
                X_rot1 = np.matmul(H1, X_in_freq)
                H2 = np.array([np.exp(1j*w*self.t_dgd/2), np.exp(-1j*w*self.t_dgd/2)])
                X_bir_freq = H2*X_rot1
                H3 = self.rot_mat(-self.theta)
                X_rot2 = np.matmul(H3, X_bir_freq)
                x_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_rot2, axes=1),axis=1), axes=1) 

                in1 = x_out[0, :]*np.exp(1j*self.gamma_nl*(8/9)*((np.abs(x_out[0, :]))**2+(np.abs(x_out[1, :]))**2)*step_s)
                in2 = x_out[1, :]*np.exp(1j*self.gamma_nl*(8/9)*((np.abs(x_out[1, :]))**2+(abs(x_out[0, :]))**2)*step_s)
        output_data = np.ravel( np.hstack( (in1, in2) ) )       
        return output_data