from nets import vgg
import tensorflow as tf
from preprocessing import vgg_preprocessing
from ..utils.upsampling import bilinear_upsample_weights

slim = tf.contrib.slim

# Mean values for VGG-16
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN


def extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping):
    """Removes the fc8 variable mapping from FCN-32s to VGG-16 model mapping dict.
    Given the FCN-32s to VGG-16 model mapping dict which is returned by FCN_32s()
    function, remove the mapping for the fc8 variable. This is done because this
    variable is responsible for final class prediction and is different for different
    tasks. Last layer usually has different size, depending on the number of classes
    to be predicted. This is why we omit it from the dict and those variables will
    be randomly initialized later.
    
    Parameters
    ----------
    vgg_16_variables_mapping : dict {string: variable}
        Dict which maps the FCN-32s model's variables to VGG-16 checkpoint variables
        names. Look at FCN-32s() function for more details.
    
    Returns
    -------
    updated_mapping : dict {string: variable}
        Dict which maps the FCN-32s model's variables to VGG-16 checkpoint variables
        names without fc8 layer mapping.
    """
    
    # TODO: review this part one more time
    vgg_16_keys = vgg_16_variables_mapping.keys()

    vgg_16_without_fc8_keys = []

    for key in vgg_16_keys:

        if 'fc8' not in key:
            vgg_16_without_fc8_keys.append(key)

    updated_mapping = {key: vgg_16_variables_mapping[key] for key in vgg_16_without_fc8_keys}
    
    return updated_mapping



def FCN_32s(image_batch_tensor,
            number_of_classes,
            is_training):
    """Returns the FCN-32s model definition.
    The function returns the model definition of a network that was described
    in 'Fully Convolutional Networks for Semantic Segmentation' by Long et al.
    The network subsamples the input by a factor of 32 and uses the bilinear
    upsampling kernel to upsample prediction by a factor of 32. This means that
    if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use 
    adapt_network_for_any_size_input(FCN_32s, 32). Note: the upsampling kernel
    is fixed in this model definition, because it didn't give significant
    improvements according to aforementioned paper.
    
    Parameters
    ----------
    image_batch_tensor : [batch_size, height, width, depth] Tensor
        Tensor specifying input image batch
    number_of_classes : int
        An argument specifying the number of classes to be predicted.
        For example, for PASCAL VOC it is 21.
    is_training : boolean
        An argument specifying if the network is being evaluated or trained.
        It affects the work of underlying dropout layer of VGG-16.
    
    Returns
    -------
    upsampled_logits : [batch_size, height, width, number_of_classes] Tensor
        Tensor with logits representing predictions for each class.
        Be careful, the output can be of different size compared to input,
        use adapt_network_for_any_size_input to adapt network for any input size.
        Otherwise, the input images sizes should be of multiple 32.
    vgg_16_variables_mapping : dict {string: variable}
        Dict which maps the FCN-32s model's variables to VGG-16 checkpoint variables
        names. We need this to initilize the weights of FCN-32s model with VGG-16 from
        checkpoint file. Look at ipython notebook for examples.
    """
    
    with tf.variable_scope("fcn_32s") as fcn_32s_scope:

        upsample_factor = 32

        # Convert image to float32 before subtracting the
        # mean pixel value
        image_batch_float = tf.to_float(image_batch_tensor)

        # Subtract the mean pixel value from each pixel
        mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

        upsample_filter_np = bilinear_upsample_weights(upsample_factor,
                                                       number_of_classes)

        upsample_filter_tensor = tf.constant(upsample_filter_np)

        # TODO: make pull request to get this custom vgg feature accepted
        # to avoid using custom slim repo.
        with slim.arg_scope(vgg.vgg_arg_scope()):

            logits, end_points = vgg.vgg_16(mean_centered_image_batch,
                                            num_classes=number_of_classes,
                                            is_training=is_training,
                                            spatial_squeeze=False,
                                            fc_conv_padding='SAME')

        downsampled_logits_shape = tf.shape(logits)

        # Calculate the ouput size of the upsampled tensor
        upsampled_logits_shape = tf.pack([
                                          downsampled_logits_shape[0],
                                          downsampled_logits_shape[1] * upsample_factor,
                                          downsampled_logits_shape[2] * upsample_factor,
                                          downsampled_logits_shape[3]
                                         ])

        # Perform the upsampling
        upsampled_logits = tf.nn.conv2d_transpose(logits,
                                                  upsample_filter_tensor,
                                                  output_shape=upsampled_logits_shape,
                                                  strides=[1, upsample_factor, upsample_factor, 1])

        # Map the original vgg-16 variable names
        # to the variables in our model. This is done
        # to make it possible to use assign_from_checkpoint_fn()
        # while providing this mapping.
        # TODO: make it cleaner
        vgg_16_variables_mapping = {}

        vgg_16_variables = slim.get_variables(fcn_32s_scope)

        for variable in vgg_16_variables:

            # Here we remove the part of a name of the variable
            # that is responsible for the current variable scope
            # original_vgg_16_checkpoint_string = variable.name[len(fcn_32s_scope.original_name_scope):-2]
            
            # Updated: changed .name_scope to .name because name_scope only affects operations
            # and variable scope is actually represented by .name
            original_vgg_16_checkpoint_string = variable.name[len(fcn_32s_scope.name)+1:-2]
            vgg_16_variables_mapping[original_vgg_16_checkpoint_string] = variable

    return upsampled_logits, vgg_16_variables_mapping