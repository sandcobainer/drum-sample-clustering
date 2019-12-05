from json import JSONEncoder
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import fnmatch
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.manifold import TSNE
import json
import pandas as pd

data = pd.DataFrame()
class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict_

def read_files(path, data):
    feature_vectors = []
    sound_paths = []

    for root, dirnames, filenames in os.walk(path):
        new_column = pd.Series(filenames)
        column_name = root.split('/')[1]
        if(column_name != ''):
            data.insert(loc=0, column = column_name, value=new_column)
            for filename in fnmatch.filter(filenames, '*.wav'):
               data[column_name + '_file'] = os.path.join(root, filename)
    data = data.dropna()

    file_cols = [col for col in data.columns if '_file' in col]
    for column in data[file_cols]:
        samples = data[column]
        for f in samples:
            y, sr = librosa.load(f)
            feat = get_features(y, sr)
            feature_vectors.append(feat)
            sound_paths.append(f)
            
    
    return data, feature_vectors, sound_paths

def get_features(y, sr):
    y = y[0:sr]  # analyze just first second
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
    feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)
    return feature_vector

    
def apply_tsne(sound_paths, feature_vectors):

    model = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.1).fit_transform(feature_vectors)
    x_axis=model[:,0]
    y_axis=model[:,1]
    print(sound_paths, x_axis,y_axis)
    plt.figure(figsize = (10,10))
    plt.scatter(x_axis, y_axis)
    plt.show()
    return x_axis, y_axis



if __name__== "__main__":

    path = 'testsamples/'
    data, feature_vectors, sound_paths = read_files(path, data)
    x_axis, y_axis = apply_tsne (sound_paths, feature_vectors)

    tsne_path = "./example-audio-tSNE.json"

    # normalize between 0 and 1
    x_norm = (x_axis - np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
    y_norm = (y_axis - np.min(y_axis)) / (np.max(y_axis) - np.min(y_axis))

    print(sound_paths, x_norm, y_norm)
    # d = [{"path":os.path.abspath(f), "point":[x, y]} for f, x, y in zip(sound_paths, x_norm, y_norm)]
    # # with open(tsne_path, 'w') as outfile:
    # #     d = json.dumps(d)
    # #     json.dump(d, outfile)

    # # print("saved %s to disk!" % tsne_path)

    # colors = cm.rainbow(np.linspace(0, 1, len(x_axis)))
    # plt.figure(figsize = (8,6))
    # plt.scatter(x_axis, y_axis, color=colors)
    # plt.show()