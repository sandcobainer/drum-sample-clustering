import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
sys.path.append('/usr/local/lib/python3/dist-packages/')

import essentia
import essentia.standard as es


def isMatch(name, patterns):
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False

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
        print(value)
        if type(value) is np.ndarray:
            value = value.tolist()
            continue
        print(i)
        result.append(value)
        i+=1
        #add_to_dict(result, keys, value)
    return result

def compute_features(complete_path):
    result = []
    # for loop over files
    for file in os.listdir(complete_path):
        if file.endswith(".wav"):
            print(file)
            # load our audio into an array
            audio = es.MonoLoader(filename=complete_path + file, sampleRate=44100)()

            # create the pool and the necessary algorithms
            pool = essentia.Pool()
            window = es.Windowing()
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
            aggrpool = es.PoolAggregator(defaultStats = [ 'mean'])(pool) #,'stdev' ])(pool)
            # write aggrpool to dict
            aggr_np = pool_to_array(aggrpool)
            print(aggr_np)
            result.append(aggr_np)
            
    return result

# See all feature names in the pool in a sorted order
# print(sorted(features.descriptorNames()))

if __name__ == '__main__':
    features = compute_features('./samples/')
    print(features)