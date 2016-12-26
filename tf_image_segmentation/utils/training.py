import tensorflow as tf


def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height, 1) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list.
    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """
    
    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    valid_entries_class_labels = class_labels[:-1]
    
    # Stack the binary masks for each class
    labels_2d = map(lambda x: tf.equal(annotation_tensor, x),
                    valid_entries_class_labels)

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)
    
    return labels_2d_stacked

def get_valid_entries_indices_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (num_valid_eintries, 2).
    Returns tensor that contains the indices of valid entries according
    to the annotation tensor. This can be used to later on extract only
    valid entries from logits tensor and labels tensor.
    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    valid_labels_indices : Tensor of size (num_valid_eintries, 2).
        Tensor with indices of valid entries
    """
    
    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    mask_out_class_label = class_labels[-1]
    
    # Get binary mask for the pixels that we want to
    # use for training. We do this because some pixels
    # are marked as ambigious and we don't want to use
    # them for trainig to avoid confusing the model
    valid_labels_mask_2d = tf.not_equal(annotation_tensor,
                                        mask_out_class_label)
    
    valid_labels_indices = tf.where(valid_labels_mask_2d)
    
    return valid_labels_indices
    