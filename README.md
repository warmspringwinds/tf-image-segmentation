# TF Image Segmentation: Image Segmentation framework

The aim of the ```TF Image Segmentation``` framework is to provide/provide a simplified way for:

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

Implemented models were tested on Restricted PASCAL VOC 2012 Validation dataset (RV-VOC12) and trained on
the PASCAL VOC 2012 Training data and additional Berkeley segmentation data for PASCAL VOC 12.
It was important to test models on restricted Validation dataset to make sure no images in the
validation dataset were seen by model during training.

This code has been used to train networks with this performance:

| Model     | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy | Model Download Link |
|-----------|-----------|---------|--------------------|----------------|---------------------|
| FCN-32s   | RV-VOC12  | 62.70   | in prog.           | in prog.       | in prog.            |
| FCN-16s   | RV-VOC12  | in prog.| in prog.           | in prog.       | in prog.            |
| FCN-8s    | RV-VOC12  | in prog.| in prog.           | in prog.       | in prog.            |



## About

The framework was developed by:

* Daniil Pakhomov