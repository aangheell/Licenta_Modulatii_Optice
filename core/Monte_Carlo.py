import numpy as np
import matplotlib.pyplot as plt

class Monte_Carlo():

    def __init__(self, chain, metric, nb_trials=100):
        self.nb_trials = nb_trials
        self.chain = chain
        self.metric = metric

    def simulate(self, param_target, param_attribute, param_values, verbose = True):
        metric_values = np.zeros((len(param_values),1))
        param_target_name = param_target.__class__.__name__

        if verbose == True:
            print("Monte Carlo Simulations ({})".format(param_target_name))

        for index in range(len(param_values)):
            metric_list = []
            value = param_values[index]
            setattr(param_target,param_attribute,value)

            for trial in range(self.nb_trials):
                self.chain.process()
                metric_list.append(self.metric.compute())
            metric_array = np.sort(np.array(metric_list))
            #metric_values[index,:] += self.metric.compute()
            metric_values[index,:] = np.mean(metric_array)

            if verbose == True:
                print("-> {}={:.4f}:\tber = {}".format(param_attribute,value,metric_values[index,:]))

        return metric_values

class Monte_Carlo_2C():

    def __init__(self, chain1, chain2, metric, nb_trials=100):
        self.nb_trials = nb_trials
        self.chain1 = chain1
        self.chain2 = chain2
        self.metric = metric

    def simulate(self, param_target1, param_target2, param_attribute, param_values, verbose = True):

        metric_values = np.zeros((len(param_values),1))
        param_target_name = param_target1.__class__.__name__

        if verbose == True:
            print("Monte Carlo Simulations ({})".format(param_target_name))

        for index in range(len(param_values)):
            value = param_values[index]
            setattr(param_target1,param_attribute,value)
            setattr(param_target2,param_attribute,value)

            for trial in range(self.nb_trials):
                self.chain1.process()
                self.chain2.process()
                metric_values[index,:] += self.metric.compute()

            metric_values[index,:] = metric_values[index,:]/self.nb_trials

            if verbose == True:
                print("-> {}={:.4f}:\tber = {}".format(param_attribute,value,metric_values[index,:]))

        return metric_values