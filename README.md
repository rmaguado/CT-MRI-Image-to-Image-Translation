# Image to image translation for CT & MRI
Medical images come in many modalities that each provide different information for physicians and researchers. Specialized research tools are oftentimes designed for one modality and incompatible with others. [Total Segmentator](https://github.com/wasserth/TotalSegmentator) is a powerful CT image segmentation model of interest. We hope to make the model functional with MRI images or other modalities by training an image to image translation model that is able to stylize MRI images as CTs.

## Project Outline

# Data structure
CTs and MRIs are broken up by slices and shuffled (each modality is kept separate at all times). They are then split into train, test and validation datasets which are each stored in a single numpy file. This file is extremely large (sometimes >100GB), DO NOT TRY to open it or read it with ```numpy.load()```. Instead use ```numpy.memmap()``` as it will only read parts of the array. 

# Trainer
The trainer class handles train and test cycles as well as updating the scheduler and logging to tensorboard. 

# Config
The config file is saved on json format to keep all the parameters that were used in training. A copy should be saved along with each model checkpoint.  

## Setup
```./setup.sh```

## Tensorboard
```tensorboard --logdir ./runs```
