import matplotlib.pyplot as plt 
import pyACA
import numpy as np
import os

def run_evaluation(complete_path_to_data_folder):
    # init
    rmsAvg = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = pyACA.ToolReadAudio(complete_path_to_data_folder + file)
            print(file)
        else:
            continue

        
    if iNumOfFiles == 0:
        return -1

    return iNumOfFiles


run_evaluation('./samples/')