import matplotlib.pyplot as plt 
import numpy as np
import os

from essentia import *
from essentia.standard import *

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
    result = np.zeros(10) 
    i=0
    # append everything to dict result
    for d in descs:
        print(i)
        value = pool[d]
        print(value)
        if type(value) is np.ndarray:
            value = value.tolist()
        result[i] = value
        i+=1
        #add_to_dict(result, keys, value)
    return result

def run(complete_path):
    # for loop over files
    for file in os.listdir(complete_path):
        if file.endswith(".wav"):
            print(file)
            # load our audio into an array
            audio = MonoLoader(filename=complete_path + file, sampleRate=44100)()

            # create the pool and the necessary algorithms
            pool = Pool()
            w = Windowing()
            spec = Spectrum()
            centroid = Centroid(range=22050)
            rolloff = R

            # compute the centroid for all frames in our audio and add it to the pool
            for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
                c = centroid(spec(w(frame)))
                pool.add('spectral.centroid', c)

            # aggregate the results (find mean if needed)
            aggrpool = PoolAggregator(defaultStats = [ 'mean'])(pool) #,'stdev' ])(pool)

            # write aggrpool to dict
            result = pool_to_array(aggrpool)w
            print(result)

            # loader = essentia.standard.MonoLoader(complete_path + file)
            # audio = loader()
            # w = Windowing(type = 'hann')
            # for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):

            # features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
            #                                   rhythmStats=['mean', 'stdev'],
            #                                   tonalStats=['mean', 'stdev'])(complete_path + file)
            # print(sorted(features['lowlevel'])

# See all feature names in the pool in a sorted order
# print(sorted(features.descriptorNames()))

if __name__ == '__main__':
    run('./samples/')