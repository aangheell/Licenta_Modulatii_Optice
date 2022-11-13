from core.Processors import Processor
import numpy as np 
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class Pilot_Extractor(Processor):
    """Extract pilots from a signal and return the input signal"""
    def __init__(self, EBL, CBL, BL, tech, config, dm=False, name="pilots_extractor"):
        self.EBL = EBL
        self.CBL = CBL
        self.BL = BL
        self.tech = tech
        self.config = config
        self.dm = dm
        self.name = name

    def process(self, input_data):
        out_data = input_data
        const = np.unique(input_data)
        if self.dm == False:
            input_data = input_data[self.EBL:self.EBL + self.CBL*self.BL]
            N = len(input_data)
            self.output_data = input_data
            data_info = np.zeros(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    #out_data[(i+1)*self.config//2+i*self.config//2] = const[i%len(const)]
                    data_info[(i+1)*self.config//2+i*self.config//2] = True
            self.data = self.output_data[data_info]
        else:
            N2 = len(input_data)//2
            in1 = input_data[:N2]
            in2 = input_data[N2:]
            input_data1 = in1[self.EBL:self.EBL + self.CBL*self.BL]
            N = len(input_data1)
            self.output_data1 = input_data1
            data_info1 = np.zeros(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    #self.output_data1[(i+1)*self.config//2+i*self.config//2] = const[i%len(const)]
                    data_info1[(i+1)*self.config//2+i*self.config//2] = True
            elif self.tech == 'LR':
                for i in range(N//self.config):
                    data_info1[(i+1)*self.config-1] = True
            self.data1 = self.output_data1[data_info1]

            input_data2 = in2[self.EBL:self.EBL + self.CBL*self.BL]
            self.output_data2 = input_data2
            data_info2 = np.zeros(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    #self.output_data2[(i+1)*self.config//2+i*self.config//2] = const[i%len(const)]
                    data_info2[(i+1)*self.config//2+i*self.config//2] = True
            elif self.tech == 'LR':
                for i in range(N//self.config):
                    data_info2[(i+1)*self.config-1] = True
            self.data2 = self.output_data2[data_info2]
            self.data = np.hstack( (self.data1, self.data2) )
            out_data = np.ravel(np.hstack((self.output_data1, self.output_data2)))

        #plt.figure()
        #plt.title('Pilots')
        #plt.subplot(211)
        #plt.plot(np.arange(len(self.data)), np.real(self.data))
        #plt.subplot(212)
        #plt.plot(np.arange(len(self.data)), np.imag(self.data))
        
        #df = pd.DataFrame({ 'N':np.arange(self.data.shape[0]).tolist(), 'y_real':np.real(self.data).tolist(), 'y_imag':np.imag(self.data).tolist() })
        #fig = make_subplots(rows=2, cols=1, subplot_titles=("Real pilots","Imaginary pilots"))
        #fig.add_trace(
        #    go.Scatter(x=np.arange(self.data.shape[0]).tolist(), y=np.real(self.data).tolist()),
        #    row=1, col=1 
        #)

        #fig.add_trace(
        #    go.Scatter(x=np.arange(self.data.shape[0]).tolist(), y=np.imag(self.data).tolist()),
        #    row=2, col=1  
        #)
        #fig.show()

        #fig2 = px.density_contour(df, x='x', y='y', title='Pilots - 2D Density')
        #fig2.update_traces(contours_coloring="fill", contours_showlabels = True)
        #fig2.show()
        return out_data

class Get_Data():
    nb_inputs = 0
    nb_outputs = 1
    data = None
    bypass = False
    def __init__(self, input_data, name="get_data"):
        self.data = input_data
        self.name = name
    
    def process(self, input_data):
        return self.data



class Get_Data_Pilots(Processor):
    def __init__(self, EBL, CBL, BL, tech, config, dm=False, out=False, qpsk=False, name="data_extractor"):
        self.EBL = EBL
        self.CBL = CBL
        self.BL = BL
        self.tech = tech
        self.config = config
        self.dm = dm
        self.out = out
        self.qpsk = qpsk
        self.name = name

    def process(self, input_data):
        out_data = input_data
        if self.qpsk:
            qpsk_symbols = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
            sigma_s = np.mean(np.abs(qpsk_symbols)**2)
            gain = 1 / np.sqrt(sigma_s)
            qpsk_symbols = qpsk_symbols * gain
        if self.dm == False:
            input_data = np.ravel(input_data)
            self.preamble = input_data[:self.EBL]
            input_data = input_data[self.EBL:self.EBL + self.CBL*self.BL]
            N = len(input_data)
            self.output_data = input_data
            data_info = np.ones(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    data_info[(i+1)*self.config//2+i*self.config//2] = False
                    if self.qpsk:
                        out_data[(i+1)*self.config//2+i*self.config//2] = np.random.choice(qpsk_symbols)
            elif self.tech == 'LR':
                for i in range(N//self.config):
                    data_info[(i+1)*self.config-1] = False
            self.data = self.output_data[data_info]
        else:
            input_data = np.ravel(input_data)
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            self.preamble1 = in1[:self.EBL]
            self.preamble2 = in2[:self.EBL]
            input_data1 = in1[self.EBL:self.EBL + self.CBL*self.BL]
            N = len(input_data1)
            self.output_data1 = input_data1
            data_info1 = np.ones(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    data_info1[(i+1)*self.config//2+i*self.config//2] = False
            elif self.tech == 'LR':
                for i in range(N//self.config):
                    data_info1[(i+1)*self.config-1] = False
            self.data1 = self.output_data1[data_info1]

            input_data2 = in2[self.EBL:self.EBL + self.CBL*self.BL]
            self.output_data2 = input_data2
            data_info2 = np.ones(N, dtype=bool)
            if self.tech == 'CME':
                for i in range(N//self.config):
                    data_info2[(i+1)*self.config//2+i*self.config//2] = False
            elif self.tech == 'LR':
                for i in range(N//self.config):
                    data_info2[(i+1)*self.config-1] = False
            self.data2 = self.output_data2[data_info2]
            self.data = np.hstack( (self.data1, self.data2) )
        if self.out:
            return self.data
        else:
            return out_data

class Symbols_Generator():
    type = "generator"
    nb_inputs = 0
    nb_outputs = 1
    data = None
    bypass = False

    def __init__(self,M=16,N_col=100, N_row=1, name="generator", **kwargs):
        self.M = M
        self.N_row = N_row
        self.N_col = N_col
        self.name = name
        if kwargs:
            for key, value in kwargs.items():
                if key=='config':
                    self.config = value
                if key=='tech':
                    self.tech = value
        else:
            self.config = False
    
    
    def process(self,input_data):
        if self.config:
            data_info = np.ones(self.N_col, dtype=bool)
            if self.tech == 'CME':
                for i in range(self.N_col//self.config):
                    data_info[(i+1)*self.config//2+i*self.config//2] = False
            elif self.tech == 'LR':
                for i in range(self.N_col//self.config):
                    data_info[(i+1)*self.config-1] = False
        self.data = np.random.randint(self.M,size=(self.N_row, self.N_col),dtype=int)
        if self.config:
            self.output_metric = self.data[:,data_info]
        return self.data
    
    def info(self):
        print("-> Processor {}".format(self.name))
        print("     parameters: {}".format(self.__dict__))