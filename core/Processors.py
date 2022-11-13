import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig
import scipy.io as sio

class Processor:
    type = "processor"
    nb_inputs = 1
    nb_outputs = 1
    bypass = False
    verbose = False

    def __init__(self,name="processor"):    
        self.name = name
    
    def process(self,input_data):
        return input_data

    def info(self):
        print("-> Processor {}".format(self.name))
        print("     bypass      {}".format(self.bypass))
        print("     parameters  {}".format(self.__dict__))

class Chain:
    bypass = False

    def __init__(self, name = "communication chain"):
        self.name = name
        self.processors = []

    def clear(self):
        self.processors = []

    def add_processor(self,processor):
        self.processors.append(processor)

    def __getattr__(self, name):
        for processor in self.processors:
            if processor.name == name:
                return processor

    def replace_processor(self,name,processor,verbose=False):
        for index in range(len(self.processors)):
            if self.processors[index].name == name:
                processor.name = name
                self.processors[index] = processor
                if verbose == True:
                    print("Processor {} changed".format(name))
    
    def process(self,data=None):

        if len(self.processors) > 0:
            for processor in self.processors:
                if processor.bypass == False:
                    data = processor.process(data)

        return data


    def info(self):
        print("........Processors Chain: {} --------".format(self.name))
        if len(self.processors) > 0:
            for processor in self.processors:
                processor.info()

class Recorder(Processor):
    
    """ Copy one part of the transmitter data stream to a particular receiver processor attribute
    """
    
    def __init__(self,name="recorder"):
        self.name=name

    def process(self,input_data):
        self.data = input_data
        return input_data


class Duplicator(Processor):

    """ Duplicate an input data stream """

    def __init__(self,source_index=[0],dest_index=[1],name="Duplicator"):
        self.source_index = source_index
        self.dest_index = dest_index
        self.name = name

    def process(self,input_data):

        output_data = input_data

        for indice in range(len(self.source_index)):
            source_index = self.source_index[indice]
            dest_index = self.dest_index[indice]
            
            duplicate_data = input_data[:,source_index]
            output_data = np.insert(output_data, dest_index, duplicate_data, axis=1)

        return output_data

class Convert_to_Matlab(Processor):
    def __init__(self,filename='data', path = "./", name="Matlab Converter"):
        self.name = name
        self.filename = filename
        self.path = path

    def process(self, input_data):
        N = len(input_data)
        end_sig = int(np.ceil(N/256) * 256)
        pad_length = np.zeros(end_sig - N)
        output_data = np.hstack( (input_data, pad_length) )
        real_data = np.real(output_data)
        imag_data = np.imag(output_data)
        sio.savemat(self.path + 'Real_{}.mat'.format(self.filename), dict(ReS=real_data))
        sio.savemat(self.path + 'Imaginary_{}.mat'.format(self.filename), dict(ImS=imag_data))
        return output_data


class Debugger(Processor):
    def __init__(self,name="debugger"):
        self.name = name

    def process(self,input_data):
        return input_data