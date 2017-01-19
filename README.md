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

The code to acquire the training and validating the model is also provided in the framework.

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|---------------------|
| FCN-32s (ours)   | RV-VOC12  | 62.70   | in prog.           | in prog.       | in prog.            |
| FCN-16s (ours)   | RV-VOC12  | 63.52   | in prog.           | in prog.       | in prog.            |
| FCN-8s (ours)    | RV-VOC12  | in prog.| in prog.           | in prog.       | in prog.            |
| FNC-32s (orig.)  | RV-VOC11  | 59.40   | 73.30              | 89.10          |                     |
| FNC-16s (orig.)  | RV-VOC11  | 62.40   | 75.70              | 90.00          |                     |
| FNC-8s  (orig.)  | RV-VOC11  | 62.70   | 75.90              | 90.30          |                     |



## About

The framework was developed by:

* Daniil Pakhomov

During implementation, some preliminary experiments and notes were reported:
- [Converting Image Classification network into FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)
- [Performing upsampling using transposed convolution](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
- [Conditional Random Fields for Refining of Segmentation and Coarseness of FCN-32s model segmentations](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/)
- [TF-records usage](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)