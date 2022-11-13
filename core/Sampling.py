import numpy as np
from numpy.lib.type_check import real
from scipy.signal import resample
import matplotlib.pyplot as plt

from core.Processors import Processor


class Upsampler(Processor):
    """Upsampling by using Fourier method of scipy"""
    def __init__(self, os, dual_mode=False, name="Upsampler"):
        self.os = os
        self.dual_mode = dual_mode
        self.name = name

    def process(self, input_data):
        if self.dual_mode == False:
            N = len(input_data)
            real_data = np.real(input_data)
            imag_data = np.imag(input_data)
            out_real = resample(real_data, self.os*N)
            out_imag = resample(imag_data, self.os*N)
            output_data = out_real + 1j*out_imag
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            real_data1 = np.real(in1)
            imag_data1 = np.imag(in1)
            out_real1 = resample(real_data1, self.os*N)
            out_imag1 = resample(imag_data1, self.os*N)
            out1 = out_real1 + 1j*out_imag1
            real_data2 = np.real(in2)
            imag_data2 = np.imag(in2)
            out_real2 = resample(real_data2, self.os*N)
            out_imag2 = resample(imag_data2, self.os*N)
            out2 = out_real2 + 1j*out_imag2
            output_data = np.ravel( np.hstack( (out1, out2) ) )

        return output_data

class Downsampler(Processor):
    """Downsampling by using Fourier method of scipy"""
    def __init__(self, os, dual_mode=False, name="Downsampler"):
        self.os = os
        self.dual_mode = dual_mode
        self.name = name

    def process(self, input_data):
        if self.dual_mode == False:
            N = len(input_data)
            real_data = np.real(input_data)
            imag_data = np.imag(input_data)
            out_real = resample(real_data, N//self.os)
            out_imag = resample(imag_data, N//self.os)
            output_data = out_real + 1j*out_imag
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            real_data1 = np.real(in1)
            imag_data1 = np.imag(in1)
            out_real1 = resample(real_data1, N//self.os)
            out_imag1 = resample(imag_data1, N//self.os)
            out1 = out_real1 + 1j*out_imag1
            real_data2 = np.real(in2)
            imag_data2 = np.imag(in2)
            out_real2 = resample(real_data2, N//self.os)
            out_imag2 = resample(imag_data2, N//self.os)
            out2 = out_real2 + 1j*out_imag2
            output_data = np.ravel( np.hstack( (out1, out2) ) )

        return output_data

class Upsampler_zero(Processor):
    """Upsampling by using zero insertion"""
    def __init__(self, os, dual_mode=False, name="Upsampler"):
        self.os = os
        self.dual_mode = dual_mode
        self.name = name

    def process(self, input_data):
        if self.dual_mode == False:
            N = len(input_data)
            output_data = np.zeros(N*self.os, dtype=complex)
            output_data[::self.os] = input_data
        else:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = np.zeros(N*self.os, dtype=complex)
            out2 = np.zeros(N*self.os, dtype=complex)
            out1[::self.os] = in1
            out2[::self.os] = in2
            output_data = np.ravel(np.hstack( (out1, out2) ))
        return output_data

class Downsampler_zero(Processor):
    """Downsampling by selecting the osth sample"""
    def __init__(self, os, dual_mode=False, name="Downsampler"):
        self.os = os
        self.dual_mode = dual_mode
        self.name = name

    def process(self, input_data):
        if self.dual_mode == False:
            output_data = input_data[::self.os]
        else:
            in1 = input_data[:len(input_data)//2]
            in2 = input_data[len(input_data)//2:]
            out1 = in1[::self.os]
            out2 = in2[::self.os]
            output_data = np.ravel(np.hstack( (out1, out2) ))
        return output_data 