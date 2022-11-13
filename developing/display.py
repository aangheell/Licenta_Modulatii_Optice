import pandas as pd 
import matplotlib.pyplot as plt
import os,sys
import numpy as np

M = 4   # ordinul modulatiei

fig = plt.figure()


for i in range(3):
    nume='fname'+str(i)
    if i==0:
        nume=os.path.join(os.getcwd(),"BER_{}QAM.csv".format(M))
        df = pd.read_csv(nume)
        x=list(df['OSNR'])
        y=list(df['BER'])
        plt.semilogy(x,y,'ro--',label="BER")
        plt.xlabel("OSNR(dB)")
        plt.ylabel("BER")
    elif i==1:
        nume=os.path.join(os.getcwd(),"BER_{}PSK.csv".format(M))
        df = pd.read_csv(nume)
        x=list(df['OSNR'])
        y=list(df['BER'])
        plt.semilogy(x,y,'bo--',label="BER")
    else:
        nume=os.path.join(os.getcwd(),"BER_{}PAM.csv".format(M))
        df = pd.read_csv(nume)
        x=list(df['OSNR'])
        y=list(df['BER'])
        plt.semilogy(x,y,'mo--',label="BER")

plt.legend(["BER_{}QAM".format(M),"BER_{}PSK".format(M),"BER_{}PAM".format(M)])  

plt.show()   
