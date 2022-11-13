import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def awgn_psk(order,snr_per_bit,type):

    gamma_b = snr_per_bit
    k = int(np.log2(order))

    if order == 2:    
        # see book Proakis "Digital communication", p 271
        argument = np.sqrt(2*gamma_b)
        value = norm.sf(argument)

    if order == 4:
        # see book Proakis "Digital communication", p 272
        argument = np.sqrt(2*gamma_b)
        term = norm.sf(argument)
        value = 2*term*(1-0.5*term)

    if order > 4:
        M = order
        argument = np.sqrt(2*k*gamma_b)*np.sin(np.pi/M)
        value = 2*norm.sf(argument)

    if type == "bin":
        value = value/k

    return value


def awgn_qam(order,snr_per_bit,type):

    gamma_b = snr_per_bit

    # see book Proakis "Digital communication", p 280
    M = order 
    k = np.log2(order)
    argument = np.sqrt(3*k*gamma_b/(M-1))
    P_sqrt_M = 2*(1-1/np.sqrt(M))*norm.sf(argument)

    value = 1-(1-P_sqrt_M)**2

    if type == "bin":
        value = value/k
    
    return value


def awgn_theo(modulation,order,snr_per_bit,type):
    if modulation == "PSK":
        value = awgn_psk(order,snr_per_bit,type)

    if modulation == "QAM":
        value = awgn_qam(order,snr_per_bit,type)

    return value

SNR_list = np.arange(0,41,2)
BER = []
for SNR in SNR_list:
    SNR_bit =  10**(SNR/10)
    BER.append(awgn_theo("QAM",16, SNR_bit/4,type="bin"))

Err = np.hstack( (np.reshape(SNR_list,(-1,1)), np.array(BER).reshape(-1,1)) ) 
np.savetxt("BER_AWQN_16.csv", Err, delimiter=',', header='OSNR,BER', comments='')

plt.figure()
plt.semilogy(SNR_list, np.array(BER))
plt.xlabel("SNR")
plt.ylabel("BER")
plt.show()