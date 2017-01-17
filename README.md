# TF Image Segmentation: CNN-based Image Segmentation framework implemented in  Tensorflow and TF-Slim library

The aim of the ```TF Image Segmentation``` framework is to provide a simplified way for:

- Converting some popular general/medical/other Image Segmentation Datasets into easy-to-use for training ```.tfrecords```
format with unified interface: different datasets but same way to store images and annotations.
- Training routine with on-the-fly data augmentation (scaling, color distortion).
- Training routine that is proved to work for particular model/dataset pair.
- Evaluating Accuracy of trained models with common accuracy measures: Mean IOU, Mean pix. accuracy, Pixel accuracy.
- Model files that were trained on a particular dataset with reported accuracy (models that were trained using
TF with reported training routine and not models that were converted from Caffe or other framework)
- Model definitions (like FCN-32s and others) that use weights initializations from Image Classification models like
VGG that are officially provided by TF-Slim library.

So far, the framework contains an implementation of the FCN models (training
and evaluation) in Tensorflow and TF-Slim library with training routine, reported accuracy,
trained models for PASCAL VOC 2012 dataset.

The end goal is to provide utilities to convert other datasets, report accuracies on them and provide models.

## PASCAL VOC 2012

While we are still tuning parameters, on the PASCAL VOC 2011
validation data subset used in the FCN paper, this code has been used
to train networks with this performance:

| Model     | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy | Model Download Link |
|-----------|-----------|---------|--------------------|----------------|---------------------|
| FCN-32s   | RV-VOC12  | 62.70   | in prog.           | in prog.       | in prog.            |
| FCN-16s   | RV-VOC12  | in prog.| in prog.           | in prog.       | in prog.            |
| FCN-8s    | RV-VOC12  | in prog.| in prog.           | in prog.       | in prog.            |



## About

This code was developed by

* Daniil Pakhomov