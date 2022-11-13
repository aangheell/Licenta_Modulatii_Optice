import numpy as np
from numpy.lib.type_check import imag
from sklearn.metrics import mean_squared_error as mse

class Binary:
    def __init__(self,M,input=None,output=None, dual_mode=False, name="BER"):
        self.M = M
        self.input = input
        self.output = output
        self.dual_mode = dual_mode
        self.name = name

    def bin_data(self,data):
        self.data = data
        self.bit = ''
        if self.M == 4:
            for i in self.data:
                self.bit += '{0:02b}'.format(i)
        elif self.M == 8:
            for i in self.data:
                self.bit += '{0:03b}'.format(i)
        elif self.M == 16:
            for i in self.data:
                self.bit += '{0:04b}'.format(i)
        elif self.M == 32:
            for i in self.data:
                self.bit += '{0:05b}'.format(i)
        elif self.M == 64:
            for i in self.data:
                self.bit += '{0:06b}'.format(i)
        elif self.M == 128:
            for i in self.data:
                self.bit += '{0:07b}'.format(i)
        elif self.M == 256:
            for i in self.data:
                self.bit += '{0:08b}'.format(i)
        elif self.M == 2:
            for i in self.data:
                self.bit += '{0:01b}'.format(i)
        return self.bit

    def compute(self):
        input_data = np.ravel(getattr(self.input,"data"))
        output_data = np.ravel(getattr(self.output,"data"))
        in_bin = self.bin_data(input_data)
        out_bin = self.bin_data(output_data)
        BER = 0
        if len(in_bin) == len(out_bin):
            for i in range(len(in_bin)):
                if in_bin[i] != out_bin[i]:
                    BER += 1
            BER /= len(in_bin)

        if self.dual_mode:
            ber_list = []
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = output_data[:N]
            out2 = output_data[N:]
            in_bin1 = self.bin_data(in1)
            in_bin2 = self.bin_data(in2)
            out_bin1 = self.bin_data(out1)
            out_bin2 = self.bin_data(out2)
            BER1 = 0
            if len(in_bin1) == len(out_bin1):
                for i in range(len(in_bin1)):
                    if in_bin1[i] != out_bin1[i]:
                        BER1 += 1
                BER1 /= len(in_bin1)
            
            BER2 = 0
            if len(in_bin2) == len(out_bin2):
                for i in range(len(in_bin2)):
                    if in_bin2[i] != out_bin2[i]:
                        BER2 += 1
                BER2 /= len(in_bin2)
            BER = [BER, BER1, BER2]
        return BER


class SER_Metric():
    def __init__(self,input_data=None,output_data=None, dual_mode=False, name="SER"):
        # The input and output object must have a data attribute
        self.input = input_data
        self.output = output_data
        self.dual_mode = dual_mode
        self.name = name
    
    def compute(self):
        input_data = np.ravel(getattr(self.input,"data"))
        output_data = np.ravel(getattr(self.output,"data"))
        N_symbols = len(input_data)
        output_data = output_data[:N_symbols]
        nb_errors = np.sum(input_data != output_data)
        SER = nb_errors / N_symbols
        if self.dual_mode == True:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = output_data[:N]
            out2 = output_data[N:]
            N_symbols1 = len(in1)
            nb_errors1 = np.sum(in1 != out1)
            SER1 = nb_errors1 / N_symbols1
            N_symbols2 = len(in2)
            nb_errors2 = np.sum(in2 != out2)
            SER2 = nb_errors2 / N_symbols2
            SER = [SER, SER1, SER2]

        return SER

    def show(self,N=10):
        input_data = getattr(self.input,"data")
        output_data = getattr(self.output,"data")
        print("--- True Values ---")
        print(input_data[:N])
        print("--- Estimated Values ---")
        print(output_data[:N])
    
    def info(self):
        print("-> Metric {}".format(self.name))
        print("   parameters: {}".format(self.__dict__))


class EVM:
    
    def __init__(self,input=None,output=None, dual_mode=False, name="EVM"):
        # The input and output object must have a data attribute
        self.input = input
        self.output = output
        self.dual_mode = dual_mode
        self.name = name
    
    def compute(self):
        input_data = np.ravel(getattr(self.input,"data"))
        output_data = np.ravel(getattr(self.output,"data"))
        Err = np.mean(np.abs(np.ravel(input_data - output_data))**2)
        EVM = 100 * np.sqrt( Err/np.mean(np.abs(np.ravel(input_data))**2) )

        if self.dual_mode == True:
            N = len(input_data)//2
            in1 = input_data[:N]
            in2 = input_data[N:]
            out1 = output_data[:N]
            out2 = output_data[N:]
            Err1 = np.mean(np.abs(np.ravel(in1 - out1))**2)
            EVM1 = 100 * np.sqrt( Err1/np.mean(np.abs(np.ravel(in1))**2) )
            Err2 = np.mean(np.abs(np.ravel(in2 - out2))**2)
            EVM2 = 100 * np.sqrt( Err2/np.mean(np.abs(np.ravel(in2))**2) )
            EVM = [EVM, EVM1, EVM2]

        return EVM

class MSE:
    def __init__(self, real_data, predicted_data, dual_mode=False, name="MSE"):
        self.real_data = real_data
        self.predicted_data = predicted_data
        self.dual_mode = dual_mode
        self.name = name

    def compute(self):
        real_data = getattr(self.real_data,"data")
        pred_data = getattr(self.predicted_data,"data")
        MSE = np.mean( (np.abs(real_data-pred_data))**2)

        if self.dual_mode == True:
            N = len(real_data)//2
            in1 = real_data[:N]
            in2 = real_data[N:]
            out1 = pred_data[:N]
            out2 = pred_data[N:]
            MSE1 = np.mean( (np.abs(in1-out1))**2)
            MSE2 = np.mean( (np.abs(in2-out2))**2)
            MSE = [MSE, MSE1, MSE2]
        return MSE