# Image to image translation for CT & MRI
Medical images come in many modalities that each provide different information for physicians and researchers. Specialized research tools are oftentimes designed for one modality and incompatible with others. (Total Segmentator)[https://github.com/wasserth/TotalSegmentator] is a powerful CT image segmentation model. We hope to make the model functional with MRI images or other modalities by training an image to image translation model that is able to stylize MRI images as CTs.

## Setup
```pip install ./requirements.txt```
### recommended
```tensorboard --logdir ./runs```