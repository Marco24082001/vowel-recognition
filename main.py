# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:06:40 2022

@author: ThanhVi
"""
import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
from automatic_processing_signal import read_result, normalize, detect_pitch_ACF, detect_threshold_mean_STE, ShortTermEnergy, SegmentSpeech, intermediate_frame, mean, stand, defindGender, defindGenderLab
import librosa
import librosa.display
from scipy import signal
from scipy.fft import rfft
import sklearn.cluster as cluster
 
def KmeanFeature(features, k):
    kmeans = cluster.KMeans(n_clusters=k, init='k-means++', random_state = 5)
    kmeans = kmeans.fit(features)
    return kmeans.cluster_centers_, kmeans
    
def getFeaturesTrain():
    threshold = 0.0700251847415527
    frame_shift = 20/1000
    frame_size = 30/1000
    FOLDER = 'NguyenAmHuanLuyen-16k'
    FOLDER_SIGNALS = next(os.walk(FOLDER))[1]
    features_cnt = []
    for FOLDER_SIGNAL in FOLDER_SIGNALS:
        features = []
        # print('Person: '+ FOLDER_SIGNAL)
        FOLDER_SIGNAL = '/'.join([FOLDER, FOLDER_SIGNAL])
        FILE_SIGNALS = os.listdir(FOLDER_SIGNAL)
        for FILE_SIGNAL in FILE_SIGNALS:
            # print('vowel: ' + FILE_SIGNAL.split('.')[0])
            X, Fs = librosa.load('/'.join([FOLDER_SIGNAL, FILE_SIGNAL]))
            X = normalize(X)
            sample_shift = int(frame_shift*Fs)
            window_length = int(frame_size*Fs)
            STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
            STE = normalize(STE)
            speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
            speech_segment = speech_segment.reshape(-1)
    
            seprate = int((speech_segment[1] - speech_segment[0])/3)
            statefull_area = speech_segment[0] + seprate
            x = X[(statefull_area*sample_shift):(statefull_area*sample_shift) + window_length*seprate]
            window = signal.windows.hamming(window_length)
            
            mfccs = librosa.feature.mfcc(x, sr=Fs, n_mfcc=13, hop_length=int(frame_size*Fs) - int(frame_shift*Fs), n_fft=int(frame_size*Fs), window = window)
            mfccs = np.mean(mfccs, axis = 1)
            features.append(mfccs)
        features_cnt.append(features)
    features_cnt = np.array(features_cnt)
    features_a = features_cnt[:,0]
    features_e = features_cnt[:,1]
    features_i = features_cnt[:,2]
    features_o = features_cnt[:,3]
    features_u = features_cnt[:,4]
    
    kmean_a, Mkmean_a = KmeanFeature(features_a, 5)
    kmean_e, Mkmean_e = KmeanFeature(features_e, 5)
    kmean_i, Mkmean_i = KmeanFeature(features_i, 5)
    kmean_o, Mkmean_o = KmeanFeature(features_o, 5)
    kmean_u, Mkmean_u = KmeanFeature(features_u, 5)
    
    return kmean_a, kmean_e, kmean_i, kmean_o, kmean_u, Mkmean_a, Mkmean_e, Mkmean_i, Mkmean_o, Mkmean_u

def euclidian_distance(x, y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance

def predict(features):
    y_predicted_a = Mkmean_a.predict(features)
    y_predicted_e = Mkmean_e.predict(features)
    y_predicted_i = Mkmean_i.predict(features)
    y_predicted_o = Mkmean_o.predict(features)
    y_predicted_u = Mkmean_u.predict(features)
    y_predicted = np.zeros((len(labels)))
    for i in range(features.shape[0]):
        dts = []
        dts.append(euclidian_distance(features[i], Mkmean_a.cluster_centers_[y_predicted_a[i]]))
        dts.append(euclidian_distance(features[i], Mkmean_e.cluster_centers_[y_predicted_e[i]]))
        dts.append(euclidian_distance(features[i], Mkmean_i.cluster_centers_[y_predicted_i[i]]))
        dts.append(euclidian_distance(features[i], Mkmean_o.cluster_centers_[y_predicted_o[i]]))
        dts.append(euclidian_distance(features[i], Mkmean_u.cluster_centers_[y_predicted_u[i]]))
        min_idx = np.argmin(dts)
        y_predicted[min_idx] += 1
    return y_predicted

def predict_vowel(feature):
    y_predicted_a = Mkmean_a.predict(feature)
    y_predicted_e = Mkmean_e.predict(feature)
    y_predicted_i = Mkmean_i.predict(feature)
    y_predicted_o = Mkmean_o.predict(feature)
    y_predicted_u = Mkmean_u.predict(feature)
    
    dts = []
    dts.append(euclidian_distance(feature[0], Mkmean_a.cluster_centers_[y_predicted_a[0]]))
    dts.append(euclidian_distance(feature[0], Mkmean_e.cluster_centers_[y_predicted_e[0]]))
    dts.append(euclidian_distance(feature[0], Mkmean_i.cluster_centers_[y_predicted_i[0]]))
    dts.append(euclidian_distance(feature[0], Mkmean_o.cluster_centers_[y_predicted_o[0]]))
    dts.append(euclidian_distance(feature[0], Mkmean_u.cluster_centers_[y_predicted_u[0]]))
    
    min_idx = np.argmin(dts)
    return labels[min_idx]
   
kmean_a, kmean_e, kmean_i, kmean_o, kmean_u, Mkmean_a, Mkmean_e, Mkmean_i, Mkmean_o, Mkmean_u = getFeaturesTrain()
labels = {0:'a', 1:'e', 2:'i', 3:'o', 4:'u'}
if __name__ == '__main__':
    threshold = 0.0700251847415527
    frame_shift = 20/1000
    frame_size = 30/1000
    N_FFT = 1024 * 2
    Fmin = 70
    Fmax = 400
    # FOLDER = 'NguyenAmHuanLuyen-16k'
    FOLDER = 'NguyenAmKiemThu-16k'
    FOLDER_SIGNALS = next(os.walk(FOLDER))[1]
    acurency = 0
    features_cnt = []
    for FOLDER_SIGNAL in FOLDER_SIGNALS:
        features = []
        # print('Person: '+ FOLDER_SIGNAL)
        FOLDER_SIGNAL = '/'.join([FOLDER, FOLDER_SIGNAL])
        FILE_SIGNALS = os.listdir(FOLDER_SIGNAL)
        for FILE_SIGNAL in FILE_SIGNALS:
            vowel = FILE_SIGNAL.split('.')[0]
            # print('vowel: ' + vowel)
            X, Fs = librosa.load('/'.join([FOLDER_SIGNAL, FILE_SIGNAL]))
            X = normalize(X)
            sample_shift = int(frame_shift*Fs)
            window_length = int(frame_size*Fs)
            w = np.hamming(window_length)
            STE = ShortTermEnergy(X, Fs, frame_size, frame_shift)
            STE = normalize(STE)
            speech, speech_segment = SegmentSpeech(X, Fs, frame_size, frame_shift, threshold)
            t = np.linspace(0, len(X)/Fs, num = len(X))
            tt = np.linspace(0, len(X)/Fs, num = len(STE))
            ttt = speech*(frame_shift*Fs)/Fs
            speech_segment = speech_segment.reshape(-1)
    
            seprate = int((speech_segment[1] - speech_segment[0])/3)
            statefull_area = speech_segment[0] + seprate
            x = X[(statefull_area*sample_shift):(statefull_area*sample_shift) + window_length*seprate]
            window = signal.windows.hamming(window_length)
            
            mfccs = librosa.feature.mfcc(x, sr=Fs, n_mfcc=13, hop_length=int(0.030*Fs) - int(0.020*Fs), n_fft=int(0.030*Fs))
            mfccs = np.mean(mfccs, axis = 1)
            features.append(mfccs)
            vowel_predicted = predict_vowel(mfccs.reshape(1, -1))
            if vowel == vowel_predicted:
                acurency += 1
            dftx = np.abs(rfft(x, N_FFT))
            dftx = np.log(dftx)
            freq = np.fft.rfftfreq(N_FFT, d=1./Fs)
            plt.figure(figsize=(10, 5))
            fig, ax = plt.subplots(3,1)
            fig.suptitle(vowel, fontsize=20)    
            
            ax[0].plot(t, X)
            ax[0].plot(tt, STE, '-')
            for i in range(tt[speech_segment].shape[0]):
                if i != 0:
                    ax[0].axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1)
                else: 
                    ax[0].axvline(tt[speech_segment][i], color='r', linestyle='-', linewidth=1, label='Biên tự động')
            
            ax[0].legend(loc='upper right')
            ax[0].set_title("Energy-based Speech/Silence discrimination")
            
            ax[1].plot(freq[:3500], dftx[:3500])
            ax[1].set_title('fft')
            font = {'family': 'serif',
                    'color':  'red',
                    'weight': 'bold',
                    'size': 15
                    }
            ax[2].plot(np.arange(mfccs.shape[0]), mfccs[:])
            ax[2].set_title('mfccs')
            
            #add text with custom font
            ax[2].text(4, 0, 'predicted: ' + vowel_predicted, fontdict=font)
            fig.tight_layout()
            # fig.legend(ax, labels=labels, loc="upper right", borderaxespad=0.1)
            fig.subplots_adjust(top=0.85)
              
            plt.show()
            
        # print(FILE_SIGNALS)
        features_cnt.append(features)
    features_cnt = np.array(features_cnt)
    features_a = features_cnt[:,0]
    features_e = features_cnt[:,1]
    features_i = features_cnt[:,2]
    features_o = features_cnt[:,3]
    features_u = features_cnt[:,4]
    
    confusion_matrix = []
    
    confusion_matrix.append(predict(features_a))
    confusion_matrix.append(predict(features_e))
    confusion_matrix.append(predict(features_i))
    confusion_matrix.append(predict(features_o))
    confusion_matrix.append(predict(features_u))
    confusion_matrix = np.array(confusion_matrix)
    print(confusion_matrix)
    print('number of correct predictions: '+ str(np.trace(confusion_matrix)))
