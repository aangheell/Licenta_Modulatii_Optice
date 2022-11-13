import numpy as np
import matplotlib.pyplot as plt
import os,sys
#sys.path.insert(0,'..\\')

sys.path.insert(0,'C:\\Users\\Admin\\Desktop\\Licenta\\First_Code\\code_licenta_as-main')

from core.Processors import Chain, Recorder
from core.Analysers import Scope
from core.Generator import Symbols_Generator, Get_Data_Pilots
from core.Modulator import Modulator, Demodulator
from core.Sampling import Upsampler, Downsampler, Upsampler_zero, Downsampler_zero
from core.Match_filtering import RRC_Resample, RRC
from core.Metrics import SER_Metric, Binary, EVM, MSE
from core.Impairments import CFO, IQ_Imbalance_albe, LPN, PMD
from core.Channels import CD_Channel, AWGN
from core.Equalization import CD_compensation, Blind_CFO, Blind_IQ, GSOP_IQ, TA_CFO, Preamble_based_cfo, Pilot_based_cpe, IQ_TX_Compensation
from core.Monte_Carlo import Monte_Carlo

### Parameters ###
M =  16                                                    # modulation order
type = "PAM"                                                # modulation type

BW = 30e9                                                   # electrical bandwidth
Ts = 1/BW                                                   # symbol period
os = 1                                                      # oversampling factor      
Fs = BW * os                                                # sampling frequency

# Chromatic dispersion parameters
D_coeff = 17e-3                                             # dispersion coefficient ps/nm-km
Lambda = 1550e-9                                            # wavelength
z = 10                                                      # fiber length

# the parameter describing chromatic dispersion
D_cum = D_coeff * z                                         # accumulated chromatic dispersion


# Carrier Frequency Offset
# CFO impact related to the sampling frequency
delta = 2*np.pi*2e8*Ts                                      # frequency offset between lasers


# Match filtering #
N_taps_mf = 60                                              # number of RRC filter taps
rollof = 0.15                                               # rolloff factor

# Laser Phase Noise #
df = 1e5                                                    # laser linewidth

# Number of polarizations
dm = False                                                   # polarization multiplexing

Nb = 500                                                    # number of symbols
Np = 300                                                    # preamble length
config = 30                                                 # pilots interval
tech = 'CME'                                                # common phase model
OSNR = 20                                                  # value of OSNR



chain = Chain()                                                                                    # The chain object
chain.add_processor(Symbols_Generator(M=M, N_row=1, N_col=Nb+Np, name="generator"))                # Symbols generator
#chain.add_processor(Get_Data_Pilots(Np, Nb, 1, tech, config, name='data_extractor'))               # Extract TX data pilots
chain.add_processor(Recorder(name="input"))                                                        # Record input data
chain.add_processor(Modulator(M=M,tip=type, normalized=False))                                           # Modulation
#chain.add_processor(Get_Data_Pilots(Np, Nb, 1, tech, config, qpsk=False, name='symbol_extractor')) # Extract TX symbols pilots
chain.add_processor(Recorder(name="symbols_in"))                                                   # Record input symbols
#chain.add_processor(Upsampler_zero(os, dual_mode=dm))                                             # Oversampling
#chain.add_processor(RRC(N_taps_mf, rollof, Ts, Fs, dual_mode=dm, pos='Tx'))                       # RRC Filter

### Impairments
#chain.add_processor(LPN(df, Ts, pos='Tx', dual=dm))                                               # Laser phase noise TX
#chain.add_processor(IQ_Imbalance_albe(1,10,1,10,dual_mode=dm))                                    # IQ imbalance TX
chain.add_processor(CD_Channel(D_cum, Lambda=Lambda, fs=Fs, dual_mode=dm))                        # Chromatic dispersion
chain.add_processor(AWGN(SNR_db=OSNR, os=os, dual_mode=dm, name='channel'))                                         # Noise
#chain.add_processor(CFO(delta, dual_mode=dm))                                                     # Carrier frequency offset
#chain.add_processor(IQ_Imbalance_albe(1,10,1,10,dual_mode=dm))                                    # IQ imbalance RX
#chain.add_processor(LPN(df, Ts, pos='Rx', dual=dm))                                               # Laser phase noise RX                                     
#chain.add_processor(Scope(type='scatter'))


### Compensation algorithms
#chain.add_processor(GSOP_IQ())                                                                                                                                     # IQ imbalance compensation (RX)
#chain.add_processor(Scope(type='scatter'))
#chain.add_processor(Preamble_based_cfo(chain.symbols_in, Np, Ts, Fs))                                                                                              # CFO compensation
#chain.add_processor(Scope(type='scatter'))
#chain.add_processor(Pilot_based_cpe(chain.symbols_in, config, preamble=Np, avg=4, avg_ml=4, ML=False, rot=True, name="ML_phase"))                    # Laser pahse noise compensation (coarse)
#chain.add_processor(Scope(type='scatter'))
#chain.add_processor(IQ_TX_Compensation(chain.symbols_in, config, preamble=Np, N_taps=1, epochs=1000, step=1e-3, name='iqtx'))                                      # IQ imbalance compensation (TX)
#chain.add_processor(Scope(type='scatter'))
#chain.add_processor(Pilot_based_cpe(chain.symbols_in, config, preamble=Np, avg=4, avg_ml=4, ML=True, rot=False, name="ML_phase"))                    # Laser phase noise compensation (finer)
#chain.add_processor(Get_Data_Pilots(0, Nb, 1, tech, config, out=True, name='receiver_extractor'))
chain.add_processor(CD_compensation(L=z,D=D_coeff,Lambda=Lambda,fs=Fs))


#chain.add_processor(RRC(N_taps_mf, rollof, Ts, Fs, dual_mode=dm, pos='Rx'))                       # RRC RX filtering
#chain.add_processor(Downsampler_zero(os, dual_mode=dm))                                           # Downsampling
#chain.add_processor(Get_Data_Pilots(0, Nb, 1, tech, config, out=True, name='receiver_extractor')) # Extract RX pilots
chain.add_processor(Recorder(name="symbols_out"))                                                  # Record output symbols
#chain.add_processor(Scope(type="scatter"))                                                         # Plot consttelation
chain.add_processor(Demodulator(M=M, tip=type, normalized=False))                                             # Demodulation
chain.add_processor(Recorder(name="output"))                                                       # Record output data
chain.process()                                                                                    # Run the chain - simulate communication

# Symbol Error Rate
metric = SER_Metric(input_data=chain.input, output_data=chain.output)
print("SER = {}".format(metric.compute()))

# Bit Error Rate
BER = Binary(M,input=chain.input, output=chain.output)
print("BER = {}".format(BER.compute()))

# Error Vector Magnitude
EVM_metric = EVM(input=chain.symbols_in,output=chain.symbols_out)
print("EVM = {}%".format(EVM_metric.compute()))


# Monte Carlo for detection #
SNR_vect = np.arange(0,31,2)
metric = Binary(M, input=chain.input,output=chain.output)
mc_ber = Monte_Carlo(chain,metric,nb_trials=1000)

Values_SNR = np.zeros((len(SNR_vect),1))
Values_SNR[:] = mc_ber.simulate(chain.channel, 'SNR_db', SNR_vect)

BER = np.matrix([SNR_vect,Values_SNR[:,0]]).T
np.savetxt("BER_{}PAM.csv".format(M),BER,delimiter=",",header="OSNR,BER",comments="")

fig = plt.figure()
plt.semilogy(SNR_vect,Values_SNR[:,0],'ro--',label="BER")
plt.xlabel("OSNR(dB)")
plt.ylabel("BER")


plt.show()                 # Command to display figures
     
