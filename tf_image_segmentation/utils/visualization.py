import numpy as np
from matplotlib import pyplot as plt


def _discrete_matshow_adaptive(data, labels_names=[], title=""):
    """Displays segmentation results using colormap that is adapted
    to a number of classes. Uses labels_names to write class names
    aside the color label. Used as a helper function for 
    visualize_segmentation_adaptive() function.
    
    Parameters
    ----------
    data : 2d numpy array (width, height)
        Array with integers representing class predictions
    labels_names : list
        List with class_names
    """
    
    fig_size = [7, 6]
    plt.rcParams["figure.figsize"] = fig_size
    
    #get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))
    
    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    
    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')
    
    plt.show()

        
def visualize_segmentation_adaptive(predictions, segmentation_class_lut, title="Segmentation"):
    """Displays segmentation results using colormap that is adapted
    to a number of classes currently present in an image, instead
    of PASCAL VOC colormap where 21 values is used for
    all images. Adds colorbar with printed names against each color.
    Number of classes is renumerated starting from 0, depending
    on number of classes that are present in the image.
    
    Parameters
    ----------
    predictions : 2d numpy array (width, height)
        Array with integers representing class predictions
    segmentation_class_lut : dict
        A dict that maps class number to its name like
        {0: 'background', 100: 'airplane'}
        
    """
    
    # TODO: add non-adaptive visualization function, where the colorbar
    # will be constant with names
    
    unique_classes, relabeled_image = np.unique(predictions,
                                                return_inverse=True)

    relabeled_image = relabeled_image.reshape(predictions.shape)

    labels_names = []

    for index, current_class_number in enumerate(unique_classes):

        labels_names.append(str(index) + ' ' + segmentation_class_lut[current_class_number])

    _discrete_matshow_adaptive(data=relabeled_image, labels_names=labels_names, title=title)
