# Interpolating-Drum MIR Semester project, Fall'19 
**document created by:** Sandeep Dasari     
**last updated:** 2019, October 28  
**last updated by:** Sandeep

### Contents
* [Overview](#overview)
* [Dependencies Overview](#dependencies)
* [Folder Contents](#folder_contents)

## <a name="overview">Overview</a> 

# Interpolating-Drum
Inspired from Google Experiments' The Infinte Drum Machine, this project seeks to take the idea of 2 Dimensional visualization of higher dimensional audio data.
Unlike t-SNE that maps high dimensional STFT data to 2-D/3-D , this project focuses on using simpler musical descriptors or popular music information retreival features like 
* Spectral centroid
* RMS
* Spectral Flux
* Attack
* Decay etc.

The reason for simplifying the dimensions is to allow the user to interactively sort a large pool of audio samples forming a general sense for these features as `filters` of sonic quality. The dataset was parsed and processed to extract features in Python. Later a Principal Component Analysis is run to reduce dimensions and perform a K-Means on the sample library. The results from PCA are sent to Max via OSC for interactive audio plotting of the one-hit samples.

## <a name="dependencies">Dependencies Overview</a>
* `Max 8` for visualization and sonofication
* `python-osc`
* `matplotlib`
* `sklearn`
* `essentia`
* `py-dub`
* `librosa`
* `pandas`
* `numpy`

## <a name="folder_contents">Folder Contents</a>

* `/samples` contains all the 160 one hit acoustic drum kit sample
* `/main.py` is the python file that parses samples and sends PCA co-ordinates to Max/MSP
* `/viz.maxpat` contains the Max patch that receives OSC, plots audio samples and plays them in real time.
* `/pca.mov` is an initial demo video of the application
