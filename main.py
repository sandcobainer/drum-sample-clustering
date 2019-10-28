import matplotlib.pyplot as plt 
import pandas as pd

import os
import sys
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import random
import time

from pythonosc import udp_client
import essentia
import essentia.standard as es


def isMatch(name, patterns):
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False

def normalize_zscore(featureData):    
    mu = np.mean(featureData,axis=1)    
    std = np.std(featureData,axis=1)
    normFeatureData = ((featureData.transpose() - mu) / std).transpose()    
    return normFeatureData

def add_to_dict(dict, keys, value):
    for key in keys[:-1]:
        dict = dict.setdefault(key, {})
    dict[keys[-1]] = value

def pool_to_array(pool, include_descs=None, ignore_descs=None):
    # a workaround to convert Pool to np.array
    # check pool descriptor names
    descs = pool.descriptorNames()
    if include_descs:
        descs = [d for d in descs if isMatch(d, include_descs)]
    if ignore_descs:
        descs = [d for d in descs if not isMatch(d, ignore_descs)]

    # let's start with 10 features
    result = []
    i = 0
    # append everything to dict result
    for d in descs:
        value = pool[d]
        
        result.append(value)
        i+=1
        #add_to_dict(result, keys, value)
    return result

def compute_features(complete_path):
    result = []
    meta_result = []
    file_count = 0
    # for loop over files
    for file in os.listdir(complete_path):
        if file.endswith(".wav"):
            file_count+=1
            # print(file +' : ' + str(file_count))

            # load our audio into an array
            audio = es.MonoLoader(filename=complete_path + file, sampleRate=44100)()

            # create the pool and the necessary algorithms
            pool = essentia.Pool()
            window = es.Windowing()
            energy = es.Energy()
            spectrum = es.Spectrum()
            centroid = es.Centroid(range=22050)
            rolloff = es.RollOff()
            crest = es.Crest()
            speak = es.StrongPeak()
            rmse = es.RMS()
            mfcc = es.MFCC()
            flux = es.Flux()
            barkbands = es.BarkBands( sampleRate = 44100)
            zerocrossingrate = es.ZeroCrossingRate()

            meta = es.MetadataReader(filename=complete_path + file, failOnError=True)()
            pool_meta, duration, bitrate, samplerate, channels = meta[7:]
            
            # centralmoments = es.SpectralCentralMoments()
            # distributionshape = es.DistributionShape()

            # compute the centroid for all frames in our audio and add it to the pool
            for frame in es.FrameGenerator(audio, frameSize = 1024, hopSize = 512):
                frame_windowed = window(frame)
                frame_spectrum = spectrum(frame_windowed)
                
                c = centroid(frame_spectrum)
                pool.add('spectral.centroid', c)

                cr = crest(frame_spectrum)
                pool.add('spectral crest', cr)

                r = rolloff(frame_spectrum)
                pool.add('spectral rolloff', r)

                sp = speak(frame_spectrum)
                pool.add('strong peak', sp)

                rms = rmse(frame_spectrum)
                pool.add('RMS', rms)

                pool.add('spectral_energy', energy(frame_spectrum))
                # (frame_melbands, frame_mfcc) = mfcc(frame_spectrum)
                # pool.add('frame_MFCC', frame_mfcc)

                fl = flux(frame_spectrum)
                pool.add('spectral flux', fl)

                # bbands = barkbands(frame_spectrum)
                # pool.add('bark bands', bbands)

                zcr = zerocrossingrate(frame_spectrum)
                pool.add('zero crossing rate', zcr)

                # frame_centralmoments = centralmoments(power_spectrum)
                # (frame_spread, frame_skewness, frame_kurtosis) = distributionshape(frame_centralmoments)
                # pool.add('spectral_kurtosis', frame_kurtosis)
                # pool.add('spectral_spread', frame_spread)
                # pool.add('spectral_skewness', frame_skewness)

            # aggregate the results (find mean if needed)
            aggrpool = es.PoolAggregator(defaultStats = ['mean'])(pool) #,'stdev' ])(pool)
            
            pool_meta.set("duration", duration)
            pool_meta.set("filename", os.path.relpath(file))

            # write pools to lists
            pool_arr = pool_to_array(aggrpool)
            result.append(pool_arr)

            meta_arr = pool_to_array(pool_meta)
            meta_result.append(meta_arr)
         
    features_df = pd.DataFrame.from_records(result)
    features_df.columns = ['centroid', 'crest','roll off','strong peak','rms','energy','flux','zcr']
    
    meta_df = pd.DataFrame.from_records(meta_result)
    meta_df.columns = ['duration','filename','metadata.tags.comment']
    del meta_df['metadata.tags.comment']

    return features_df,meta_df

# See all feature names in the pool in a sorted order
# print(sorted(features.descriptorNames()))

if __name__ == '__main__':
    features,metadata = compute_features('./samples/')
   
    # normalized_features = preprocessing.normalize(features,axis =1)
    features = preprocessing.StandardScaler().fit_transform(features)
    standardized_features = pd.DataFrame(features, columns = ['centroid', 'crest','roll off','strong peak','rms','energy','flux','zcr'])
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(standardized_features)
    principalComponents = preprocessing.MinMaxScaler().fit_transform(principalComponents)
    
    principal_df = pd.DataFrame(data = principalComponents
             ,columns = ['pc_1', 'pc_2'])
    
    print(principal_df.head())
    kmeans = KMeans(n_clusters=4).fit(principal_df)
    centroids = kmeans.cluster_centers_
    plt.scatter(principal_df['pc_1'], principal_df['pc_2'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    df = pd.concat([principal_df,metadata],axis = 1)
    df = df.sort_values('filename')

    client = udp_client.SimpleUDPClient('127.0.0.1', 7400)

    count = 1
    for index,row in df.iterrows():
        print( count, "pca." + str(count), row['filename'], row['pc_1'], row['pc_2'])
        client.send_message("/pca", [ count, "pca."+str(count), row['pc_1']*500, row['pc_2']*500,row['filename'] ] )
        count+=1

    plt.show()
   
