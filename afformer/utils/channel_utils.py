import numpy as np

def compute_pl_v2i_nlos_db(d, f_GHz, h):
    pl_los = 32.4 + 21*np.log10(d) + 20*np.log10(f_GHz)
    return max(pl_los, 22.4 + 35.3*np.log10(d) + 21.3*np.log10(f_GHz)-0.3*(h-1.5))

def compute_pl_v2i_los_db(d, f_GHz):
    return 32.4 + 21*np.log10(d) + 20*np.log10(f_GHz)

def compute_pl_v2v_nlos_db(d, f_GHz):
    pl_los = 38.77+16.7*np.log10(d) + 18.2*np.log10(f_GHz)   
    return max(pl_los, 36.85+30*np.log10(d) + 18.9*np.log10(f_GHz))

def compute_pl_v2v_los_db(d, f_GHz):
    return 38.77+16.7*np.log10(d) + 18.2*np.log10(f_GHz)

def fspl(d, f_GHz):
    return 20*np.log10(d) + 20*np.log10(f_GHz*1e9) + 20*np.log10(4*np.pi/3e8)

def winner_nlos(d1, d2, f_GHz):
    d1 = max(d1, 1.0); d2 = max(d2, 1.0)
    pl_LOS_1 = 22.7*np.log10(d1) + 41.0 + 20*np.log10(f_GHz/5.0)
    pl_LOS_2 = 22.7*np.log10(d2) + 41.0 + 20*np.log10(f_GHz/5.0)
    n1 = max(2.8 - 0.0024*d1, 1.84)
    n2 = max(2.8 - 0.0024*d2, 1.84)
    d12 = max(d1 + d2, 1.0)
    pl_12 = pl_LOS_1 + 20 - 12.5*n1 + 10*n1*np.log10(d12) + 3*np.log10(f_GHz/5.0)
    pl_21 = pl_LOS_2 + 20 - 12.5*n2 + 10*n2*np.log10(d12) + 3*np.log10(f_GHz/5.0)
    return min(pl_12, pl_21)

def winner_los(d, f_GHz):
    return 22.7*np.log10(d) + 41.0 + 20*np.log10(f_GHz/5.0)

def rician_h_ss(K_dB: float):
    """Complex Rician fading coefficient h_ss with E[|h_ss|^2]=1."""
    K = 10**(K_dB / 10.0)  # linear
    phi = np.random.uniform(0, 2*np.pi)
    h_LOS = np.exp(1j * phi)
    h_NLOS = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    h_ss = np.sqrt(K/(K+1)) * h_LOS + np.sqrt(1/(K+1)) * h_NLOS
    return h_ss

def rayleigh_h_ss():
    """Complex Rayleigh fading coefficient h_ss with E[|h_ss|^2]=1."""
    return (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)