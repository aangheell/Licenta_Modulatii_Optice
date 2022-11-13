import numpy as np
import matplotlib.pyplot as plt
import os,sys
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

### Parameters ###
M = 2                                                       # modulation order
type = "QAM"                                                # modulation type

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

Nb = 300                                                    # number of symbols
Np = 300                                                    # preamble length
config = 30                                                 # pilots interval
tech = 'CME'                                                # common phase model
OSNR = 20                                                   # value of OSNR

chain = Chain()                                                                                    # The chain object
chain.add_processor(Symbols_Generator(M=M, N_row=1, N_col=Nb, name="generator"))                # Symbols generator
#chain.add_processor(Get_Data_Pilots(Np, Nb, 1, tech, config, name='data_extractor'))               # Extract TX data pilots
#chain.add_processor(Recorder(name="input"))        
#chain.add_processor(Modulator(M=M, normalized=True)) 
#chain.add_processor(Scope(type="scatter"))       
x=chain.process()  
print(x)

