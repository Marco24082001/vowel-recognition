# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:18:59 2021

@author: ThanhVi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft
from scipy.signal import find_peaks
import math
import os
from scipy.signal import hilbert

# mormalize data from [-1, 1]
def normalize(X):
    X = X.reshape(-1,)
    return X/X[np.argmax(np.fabs(X))]

# get median
def median(x):
    midx = int(x.shape[0]/2)    # middle point
    x= np.sort(x)   # sort ascending
    return x[midx]  # return middle value

#  median filter
def median_filter(x):
    n = len(x)  # length of x
    # for i in range(2, n-2): 
    #     x[i] = median(x[i-2:i+3])   # select slide window = 5
        
    # for i in range(0, 2):
    #     x[i] = median(x[i:i+5])
        
    # for i in range(n-2, n):
    #     x[i] = median(x[i-4: i+1])
        
    for i in range(3, n-3): 
        x[i] = median(x[i-3:i+4])   # select slide window = 7
        
    for i in range(0, 3):
        x[i] = median(x[i:i+7])
        
    for i in range(n-3, n):
        x[i] = median(x[i-6: i+1])
    return x

# get mean data
def mean(x):
    return np.sum(x)/x.shape[0]

# get standard deviation data
def stand(x):
    y = mean(x)
    return np.sqrt(np.sum((x-y)**2)/(x.shape[0]))

# get short term energy of signal
def ShortTermEnergy(X, Fs, frame_size, frame_shift):
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    STE = []
    summ = 0
    for i in range(math.floor(len(X)/sample_shift) - math.ceil(window_length/sample_shift)):
        summ = np.sum(X[(i*sample_shift):(i*sample_shift) + window_length]**2)
        STE.append(summ)
    return np.array(STE).astype(float)
    
# get frame speech
def SegmentSpeech(X, Fs, frame_size, frame_shift, threshold):
    STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
    STE = normalize(STE)
    SEG = np.where(STE > threshold, 1, 0)
    # 0 0 0 > 300 ms silence else speech => 300/20 = 15 frame
    # k = 300/frame_size

    for i in range(6, len(STE)-6):
        if np.any(SEG[i-6:i] == 1) and np.any(SEG[i:i+7] == 1):
            SEG[i] = 1
            
    # for e in range(15, len(STE)-15):
    #     if np.any(SEG[e-15:e] == 0) and np.any(SEG[e:e+16] == 0):
    #         SEG[e] = 0
            
    speech_segment = [] 
    speech = []
    for j in range(len(SEG) - 1):
        if j == 0:
            if SEG[j] == 1:
                speech_segment.append(j)
        if (SEG[j] == 0 and SEG[j + 1] == 1) or (SEG[j] == 1 and SEG[j+1] == 0):
            speech_segment.append(j+1)
        if (SEG[j] == 1):
            speech.append(j)
    
    # print(speech_segment)
    if len(speech_segment)%2 != 0:
        speech_segment = speech_segment[: -1]
        
    tmp = np.array(speech_segment).reshape((-1, 2))
    speech_segment = []
    for segment in tmp:
        if segment[1] - segment[0] > 6:
            speech_segment.append(segment)
    
    return np.array(speech), np.array(speech_segment).reshape((-1, 2))

# get region speech, silence, F, Fst from file .lab
def read_result(x):
    f = open(x, 'r', encoding = 'utf-8')
    r = f.readlines()
    r = [s.replace('\n', ' ') for s in r]
    r = [s.replace('\t', ' ') for s in r]
    r = [s.split(' ') for s in r]
    [s.remove('') for s in r if len(s)!=2]
    
    F = np.array(r[:-3:-1]) # frequency standard and mean
    V = np.array(r[:-2])
    v = np.delete(V, np.where(V[:,2]=='sil'), axis=0)[:,:-1]  # voice region
    sil = np.delete(V, np.where(V[:,2]!='sil'), axis=0)[:,:-1]  # silence region 
    return v.astype(float), sil.astype(float), F

# detect threshold by binary search
def detect_threshold_STE(X, Y, Fs, frame_size, frame_shift):
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
    STE = normalize(STE)
    v, s, f = read_result(Y)
    v = ((v*Fs)/(sample_shift)).astype(int).reshape(-1)
    s = ((s*Fs)/(sample_shift)).astype(int).reshape(-1)
    s[-1] = s[-1] - window_length/ sample_shift - 1
    
    # initial f and g
    # f contains lower band
    # g contains upper band
    f = np.array([], dtype=int)
    g = np.array([], dtype=int)
    
    for i in range(int(s.size/2)):  
        f = np.append(f, np.arange(s[i*2], s[i*2 + 1] + 1))
    for i in range(int(v.size/2)):
        g = np.append(g, np.arange(v[i*2], v[i*2 + 1] + 1))
    f = STE[f]
    g = STE[g]
    
    Tmin = np.min(np.array([np.min(f), np.min(g)]))
    Tmax = np.max(np.array([np.max(f), np.max(g)]))
    
    j = -1
    q = -1
    value = 0
    while True:
        Tmid = (Tmax + Tmin)/2
        i = sum(1 for ste in f if ste < Tmid)
        p = sum(1 for ste in g if ste > Tmid)
        if i != j or p != q:
            value = np.sum(np.maximum(f - Tmid, 0))/f.size - np.sum(np.maximum(Tmid - g, 0))/g.size
        else:
            break
        if value > 0:
            Tmin = Tmid
        else:
            Tmax = Tmid
        j = i
        q = p
    return Tmid

# get threshold from 4 train_signal
def detect_threshold_mean_STE(frame_size, frame_shift):
    PATH_X = 'train_signal-44k/X'
    PATH_Y = 'train_signal-44k/Y'
    FILE_X = os.listdir(PATH_X)
    FILE_Y = os.listdir(PATH_Y)
    thresholds = np.array([])
    print('\n\n')
    for i in range(len(FILE_X)):
        Fs, X = wavfile.read(PATH_X + '/' + FILE_X[i])
        X = normalize(X)
        threshold = detect_threshold_STE(X, PATH_Y + '/' + FILE_Y[i], Fs, frame_size, frame_shift)
        thresholds = np.append(thresholds, threshold)
        print('Ngưỡng file ' + FILE_X[i] + ': ', threshold)
    thresholds_mean = mean(thresholds)
    thresholds_std = stand(thresholds)
    min_thresholds = thresholds_mean - thresholds_std
    max_thresholds = thresholds_mean + thresholds_std
    print('Ngưỡng trung bình: ', thresholds_mean)
    print('Độ lệch chuẩn của ngưỡng: ', thresholds_std)
    print('Ngưỡng được chọn: ' + str(min_thresholds) + ' - ' + str(max_thresholds))
    print('\n\n')
    return thresholds_mean

# analyze ste from 4 file test_signal 
def analytics_STE(frame_size, frame_shift):
    PATH_X = 'test_signal/X'
    PATH_Y = 'test_signal/Y'
    FILE_X = os.listdir(PATH_X)
    FILE_Y = os.listdir(PATH_Y)
    for i in range(len(FILE_X)):
        Fs, X = wavfile.read(PATH_X + '/' + FILE_X[i])
        sample_shift = int(frame_shift*Fs)
        X = normalize(X)
        v, s, F = read_result(PATH_Y + '/' + FILE_Y[i])
        v = ((v*Fs)/(sample_shift)).astype(int)
        s = ((s*Fs)/(sample_shift)).astype(int)
        # s[-1] = s[-1] - window_length/ sample_shift - 1
        STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
        STE = normalize(STE)
        # print(STE)
        # print(s)
        
        v_ste = np.array([])
        s_ste = np.array([])
        for j in range(v.size):
            v_ste = np.append(v_ste, STE[v[i][0]: v[i][1]])
        for j in range(s.size):
            s_ste = np.append(s_ste, STE[s[i][0]: s[i][1]])
        # v_ste = STE[v]
        # s_ste = STE[s]
        
        v_ste_mean = stand(v_ste)
        s_ste_mean = stand(s_ste)
        print('mean short term energy of voice: ', v_ste_mean)
        print('mean short term energy of silence: ', s_ste_mean)
        
        

def findPeak(dfft, idx):
    n = dfft.size
    for i in range(idx, n-4):
        if np.argmax(dfft[i:i+5]) == 2:
            if dfft[i + 1] > dfft[i] and dfft[i + 4] < dfft[i + 3]:
                return i + 3

def ACF(X, W, t, lag):
    # Autocorrelation
    # return np.sum(
    #         X[t : t + W - 1 - lag] *
    #         X[lag + t : lag + t + W - 1 - lag]
    #         )
    
    # Normalize autocorrelation
    return np.sum(
            X[t : t + W - 1 - lag] *
            X[lag + t : lag + t + W - 1 - lag]
            )/np.sqrt((np.sum(X[t : t + W - 1 - lag]**2))*(np.sum(X[lag + t : lag + t + W - 1 - lag]**2)))
   
# get envelope spectrum
def envelope_spectrum(peaks, dftx):
    # dftx: spectrum from rfft frame signal*w
    # peaks: peaks of spectrum from dftx
    
    u_x = np.array([0])
    u_y = np.array(dftx[0])
    u_x = np.append(u_x,peaks)
    u_y = np.append(u_y,dftx[peaks])
    u_x = np.append(u_x, dftx.size - 1)
    u_y = np.append(u_y, 0)
    
    # use interpolation to get envelope
    env = np.interp(np.arange(dftx.size), u_x, u_y)
    return env

def spectral_envelope(dftx):
    
    pass

def detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech_segment, N_FFT):
    resolution = Fs/N_FFT
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    w = np.hamming(window_length)   # window hamming
    freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
    F00 = []
    F0 = 0
    
    for i in range(len(speech_segment)):
        x = X[(speech_segment[i]*sample_shift):(speech_segment[i]*sample_shift) + window_length] * w
        bounds = [0, int(Fmax/resolution) * 2]
        dftx = np.abs(rfft(x, N_FFT))
        dftx = normalize(dftx)
        # dfft = np.copy(dftx[:])
        
        # Compute crossing-rate spectrum
        peaks_dftx = find_peaks(dftx,height= 0.02 ,distance=Fmin/resolution+1, prominence = 0.03)
        np.diff(peaks_dftx[0])
        # envelope spectrum
        env = envelope_spectrum(peaks_dftx[0], dftx)        
        g = np.sum(dftx * env)/np.sum(env * env)
        dftx = dftx - g*env
        # apply normalize Autocorrelation
        ACF_vals = normalize(np.array([ACF(dftx, dftx.size, 0, l) for l in range(*bounds)]))
        peaks_ACF = find_peaks(ACF_vals)

        if len(peaks_ACF[0]) != 0:
            F0 = freq[peaks_ACF[0][0]] 
        else:
            F0 = freq[0]
        F00.append([speech_segment[i], F0])
        
        # plot
        # plt.figure(figsize=(30, 15))
        # plt.subplot(2, 1, 1)
        # l1, = plt.plot(freq[:800], dftx[:800], label= 'sdfsd')
        # l2, = plt.plot(freq[:800], dfft[:800], '-')
        # l3, = plt.plot(freq[:800], env[:800])
        # plt.legend((l1,l2,l3), ["zero-crossing spectrum", "spectrum", "envelope spectrum"], prop= {'size': 15})
        # plt.title("Analytics spectrum", fontsize = 15)
        # plt.xlabel("Hz", fontsize = 15)
        # plt.ylabel("Magnitude", fontsize = 15)
        # plt.subplot(2, 1, 2)
        # plt.plot(np.arange(ACF_vals.size), ACF_vals)
        # plt.plot(np.arange(ACF_vals.size)[peaks_ACF[0]], ACF_vals[peaks_ACF[0]], 'ro')
        # plt.title("Spectral Autocorrelation", fontsize = 5)
        # plt.xlabel("Lag", fontsize = 15)
        # plt.ylabel("Magnitude", fontsize = 15)
        
    F00 = np.array(F00)
    F00[:,1] = median_filter(F00[:,1])
        
    return np.array(F00)

# get a sample fft to plot
def intermediate_frame(X, Fs, N_FFT, frame_size, frame_shift, speech_segment, Fmin, Fmax):
    resolution = Fs/N_FFT
    sample_shift = int(frame_shift*Fs)
    window_length = int(frame_size*Fs)
    w = np.hamming(window_length)   # window hamming
    freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
    x = X[(speech_segment[13]*sample_shift):(speech_segment[13]*sample_shift) + window_length] * w
    bounds = [0, int(Fmax/resolution) * 2]
    dftx = np.abs(rfft(x, N_FFT))
    dftx = normalize(dftx)
    dfft = np.copy(dftx[:])
    
    peaks_dftx = find_peaks(dftx,height= 0.02 ,distance=Fmin/resolution+1, prominence = 0.03)
    np.diff(peaks_dftx[0])
    env = envelope_spectrum(peaks_dftx[0], dftx)
    g = np.sum(dftx * env)/np.sum(env * env)
    dftx = dftx - g*env
        
    ACF_vals = normalize(np.array([ACF(dftx, dftx.size, 0, l) for l in range(*bounds)]))
    peaks_ACF = find_peaks(ACF_vals)
    
    return freq, dftx, dfft, env, ACF_vals, peaks_ACF

def defindGender(F):
    if F >= 70 and F <= 150:
        return 'male'
    elif F > 150:
        return 'female'
    else:
        return 'undefinded'
    
def defindGenderLab(nameFile):
    if nameFile[2] == 'M':
        return 'male'
    else:
        return 'female'
if __name__ == '__main__':
    
    frame_shift = 10/1000
    frame_size = 20/1000
    threshold = 0.0024
    N_FFT = 1024*2
    Fmin = 70
    Fmax = 400
        
    
    Fs = 10000
    F1 = 200
    F2 = 100
    duration = 5
    nSamples = 100
    
    t = np.linspace(0, 2, num = 2*Fs)
    # Fs, X = wavfile.read('TinHieuHuanLuyen/01MDA.wav') # 135.5
    # Fs, X = wavfile.read('TinHieuHuanLuyen/02FVA.wav') # 239.7
    # Fs, X = wavfile.read('TinHieuHuanLuyen/03MAB.wav') # 115.0
    # Fs, X = wavfile.read('TinHieuHuanLuyen/06FTB.wav') # 202.9
    
    # Fs, X = wavfile.read('TinHieuKiemThu/30FTN.wav') # 233.2
    # Fs, X = wavfile.read('TinHieuKiemThu/42FQT.wav') # 242.7
    # Fs, X = wavfile.read('TinHieuKiemThu/44MTT.wav') # 125.7
    Fs, X = wavfile.read('TinHieuKiemThu/45MDV.wav') # 177.8
    
    # Y = 'TinHieuHuanLuyen/01MDA.lab'
    # Y = 'TinHieuHuanLuyen/02FVA.lab'
    # Y = 'TinHieuHuanLuyen/03MAB.lab'
    # Y = 'TinHieuHuanLuyen/06FTB.lab'
    
    # Y = 'TinHieuKiemThu/30FTN.lab'
    # Y = 'TinHieuKiemThu/42FQT.lab'
    # Y = 'TinHieuKiemThu/44MTT.lab'
    Y = 'TinHieuKiemThu/45MDV.lab'
    
    # Y = 'TinHieuKiemThu/42FQT.lab'
    # X = normalize(X)
    # print(detect_threshold_STE(X, Y, Fs, frame_size, frame_shift))
    detect_threshold_mean_STE(frame_size, frame_shift)
    # T = 1/Fs
    # speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
    # STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
    # STE = normalize(STE)
    
    # analytics_STE(frame_size, frame_shift)
        
    # v, sil, F = read_result(Y)
    # v = v.astype(float) * Fs
    # v = (v/(frame_shift*Fs)).astype(int).reshape(-1)
    # F00 = detect_pitch_ACF(X, Fs, Fmin, Fmax, frame_size, frame_shift, speech, N_FFT)
    # F00[:,0] = F00[:,0]*frame_shift
    # Fmean = mean(F00[:,1])
    
    # print('result: ')
    # print(Fmean)
    
    # freq, dftx, dfft, env, ACF_vals, peaks_ACF = intermediate_frame(X, Fs, N_FFT, frame_size, frame_shift, speech, Fmin, Fmax)
    # _ = np.zeros((len(X)))
    
    # t = np.linspace(0, len(X)/Fs, num = len(X))
    # tt = np.linspace(0, len(X)/Fs, num = len(STE))
    # ttt = speech*(frame_shift*Fs)/Fs
    # plt.figure(figsize=(40, 40))
    # plt.subplot(2, 2, 1)
    # plt.title("Energy-based Speech/Silence discrimination")
    # plt.plot(t, X)
    # plt.plot(tt, STE, '-')
    # # plt.plot(ttt, F00/100, '.')
    # for i in range(tt[speech_segment].shape[0]):
    #     plt.axvline(tt[speech_segment][i], color='r', linestyle=':', linewidth=2)
    # for i in range(tt[v].shape[0]):
    #     plt.axvline(tt[v][i], color='b', linestyle=':', linewidth=2)
    
    # ax1 = plt.subplot(2, 2, 3)
    # ax1.plot(t, _)
    # ax1.set_ylim([0,400])
    # ax1.plot(F00[:,0], F00[:,1], 'r.')
    
    # plt.title("frequency")
    # plt.xlabel("Time")
    # plt.ylabel("Hz")
    
    # plt.subplot(2, 2, 2)
    # l1, = plt.plot(freq[:1500], dftx[:1500], label= 'sdfsd')
    # l2, = plt.plot(freq[:1500], dfft[:1500], '-')
    # l3, = plt.plot(freq[:1500], env[:1500])
    
    # plt.legend((l1,l2,l3), ["zero-crossing spectrum", "spectrum", "envelope spectrum"], prop= {'size': 15})
    # plt.title("Analytics spectrum")
    # plt.xlabel("Hz")
    # plt.ylabel("Magnitude")
    # plt.subplot(2, 2, 4)
    # plt.plot(np.arange(ACF_vals.size), ACF_vals)
    # plt.plot(np.arange(ACF_vals.size)[peaks_ACF[0]], ACF_vals[peaks_ACF[0]], 'ro')
    # plt.title("Spectral Autocorrelation")
    # plt.xlabel("Lag")
    # plt.ylabel("Magnitude")
    # plt.show()
    
    
    
    
    
    
    
    # tt = np.linspace(0, len(X)/Fs, num = len(STE))
    # ttt = speech*(frame_shift/1000*Fs)/Fs
    
    # plt.figure(figsize=(50, 20))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, X)
    # plt.plot(tt, STE, '-')
    # plt.plot(ttt, F00/200, '.')
    
    # for i in range(tt[speech_segment].shape[0]):
    #     plt.axvline(tt[speech_segment][i], color='b', linestyle=':', linewidth=2)
    # plt.show()
    # PlotfftFram(X, Fs, frame_size, frame_shift)




