from operator import le
import os
from numpy.core.numeric import outer
from numpy.core.records import array
import pandas as pd 
import numpy as np
from scipy.signal.filter_design import normalize
from core.Processors import Processor
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import firls
import seaborn as sns


class Modulator(Processor):
    def __init__(self,M = 4, normalized = True, gain = 1, name = "QAM Modulator", tip='QAM',**kwargs):
        self.normalized = normalized
        self.gain = gain
        self.name = name
        self.tip=tip
        
        if tip=='OOK':
            self.type=tip
            self.M=2
        elif tip=='BASK':
            self.type=tip
            self.M=2
        elif tip=='QAM':
            self.M=M
            self.type = "{}QAM".format(self.M)
            self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
            self.df = pd.read_csv(self.filename)
        elif tip=='PSK':
            self.M=M
            if self.M==4:
                self.type = "{}QAM".format(self.M)
                self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
                self.df = pd.read_csv(self.filename)
            elif self.M==8 or self.M==16:
                self.type = "{}QPSK".format(self.M)
                self.filename = os.path.join("../csv","{}QPSK.csv".format(self.M))
                self.df = pd.read_csv(self.filename)
        elif tip=='PAM':
            self.M=M
            self.type = "{}PAM".format(self.M)
        else:
            exit('Eroare de introducere de tip')

        if kwargs:
            for key, value in kwargs.items():
                if key=='config':
                    self.config = value
                if key=='tech':
                    self.tech = value
        else:
            self.config = False

    def constellation(self):
        

        if self.type=='OOK':
            simbol = [0,1]
            symbol=[]
            for i in simbol:
                symbol.append(np.complex(i))
            self.symbol = np.array(symbol)
        elif self.type=='BASK':
            simbol = [-1,1]
            symbol=[]
            for i in simbol:
                symbol.append(np.complex(i))
            self.symbol = np.array(symbol)
        elif self.type.find('QAM')!= -1:
            symbol=[]
            for i in self.df.columns:
                symbol.append(np.complex(i.replace("i","j")))
            self.symbol=np.array(symbol)
        elif self.type.find('PSK')!=-1:
            symbol=[]
            for i in self.df.columns:
                symbol.append(np.complex(i.replace("i","j")))
            self.symbol=np.array(symbol)
        elif self.type.find('PAM')!=-1:
            simbol=np.linspace(-1,1,self.M)
            symbol=[]
            for i in simbol:
                symbol.append(np.complex(i))
            self.symbol=np.array(symbol)


        if self.normalized == True:
            self.sigma_s = np.mean(np.abs(self.symbol)**2)
            self.gain = 1 / np.sqrt(self.sigma_s)
        self.symbol = self.symbol * self.gain
        return self.symbol
    
    def get_pilots(self, s):
        N = len(s)
        s_vect = np.reshape(s, (N,1), order='F')
        s_vect_ext = np.vstack((np.real(s_vect), np.imag(s_vect)))
        G = np.zeros((N//self.config,N))
        data_info = np.ones(N, dtype=bool)
        if self.tech == 'CME':
            for i in range(N//self.config):
                G[i,(i+1)*self.config//2+i*self.config//2] = 1
                data_info[(i+1)*self.config//2+i*self.config//2] = False
        elif self.tech == 'LR':
            for i in range(N//self.config):
                G[i,(i+1)*self.config-1] = 1
                data_info[(i+1)*self.config-1] = False
        G_pilots_1 = np.hstack((G,np.zeros((N//self.config,N))))
        G_pilots_2 = np.hstack((np.zeros((N//self.config,N)),G))
        G_pilots = np.vstack((G_pilots_1, G_pilots_2))
        s_pilots = np.matmul(G_pilots,s_vect_ext)
        data_info = np.reshape(data_info,(N,1), order='F')
        output_metric = s_vect[data_info]
        return (s_pilots, G_pilots, output_metric, data_info)

    def process(self,input_data):
        self.data = input_data
        output_data = np.zeros_like(input_data, dtype=complex)
        constellation = self.constellation()
        for j in range(self.data.shape[0]):
            symb = []
            for symbol in self.data[j,:]:
                symb.append(constellation[symbol])
            output_data[j,:] = np.array(symb)[:]
        if self.config:
            self.s_pilots, self.G_pilots, self.output_metric, self.data_info = self.get_pilots(output_data)
        if output_data.shape[0]==1:
            output_data = np.ravel(output_data)
        return output_data

class Demodulator(Modulator):
    def __init__(self, M = 16, normalized = True, gain = 1, tip='QAM', name = "QAM Demodulator"):
        self.normalized = normalized
        self.gain = gain
        self.name = name
        self.symbol = []
        self.symb = []


        if tip=='OOK':
            self.type=tip
            self.M=2
        elif tip=='BASK':
            self.type=tip
            self.M=2
        elif tip=='QAM':
            self.M=M
            self.type = "{}QAM".format(self.M)
            self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
            self.df = pd.read_csv(self.filename)
        elif tip=='PSK':
            self.M=M
            if self.M==4:
                self.type = "{}QAM".format(self.M)
                self.filename = os.path.join("../csv","{}QAM.csv".format(self.M))
                self.df = pd.read_csv(self.filename)
            elif self.M==8 or self.M==16:
                self.type = "{}QPSK".format(self.M)
                self.filename = os.path.join("../csv","{}QPSK.csv".format(self.M))
                self.df = pd.read_csv(self.filename)
        elif tip=='PAM':
            self.M=M
            self.type = "{}PAM".format(self.M)
        else:
            exit('Eroare de introducere de tip')


    def process(self,input_data):
        constellation = self.constellation()
        output_data = np.zeros_like(input_data, dtype=np.int)
        if input_data.ndim == 2:
            for l in range(input_data.shape[0]):
                distances = np.abs(np.transpose([input_data[l,:]])-constellation)**2
                output_data[l,:] = np.argmin(distances,1)
        else:
            distances = np.abs(np.transpose([input_data[:]])-constellation)**2
            output_data[:] = np.argmin(distances,1)
        return output_data

    def bin_data(self,input_data):
        self.bit = ''
        if self.M == 4:
            for i in input_data:
                self.bit += '{0:02b}'.format(i)
        elif self.M == 8:
            for i in input_data:
                self.bit += '{0:03b}'.format(i)
        elif self.M == 16:
            for i in input_data:
                self.bit += '{0:04b}'.format(i)
        elif self.M == 32:
            for i in input_data:
                self.bit += '{0:05b}'.format(i)
        elif self.M == 64:
            for i in input_data:
                self.bit += '{0:06b}'.format(i)
        elif self.M == 128:
            for i in input_data:
                self.bit += '{0:07b}'.format(i)
        elif self.M == 256:
            for i in input_data:
                self.bit += '{0:08b}'.format(i)
        return self.bit