"""
Instantaneous phase synchrony using the Hilbert transform

Adapted from code originally developed by Jin Hyun Cheong
"""
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_instaneous_sync(d1, d2, lowcut=.01, highcut=.5, fs = 30., order=1):
    y1 = butter_bandpass_filter(d1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    y2 = butter_bandpass_filter(d2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    al1 = np.angle(hilbert(y1),deg=False)
    al2 = np.angle(hilbert(y2),deg=False)
    phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
    phase_synchrony = 1 - np.sin(np.abs(al1-al2)/2)
    return phase_synchrony

def sync_average(d1, d2, **kwargs):
    ts = find_instaneous_sync(d1, d2, **kwargs)
    return np.mean(ts)