import sys
import tensorflow as tf
from ..utils.upsampling import bilinear_upsample_weights


# Using custom slim repository
sys.path.append("/home/dpakhom1/workspace/my_models/slim/")
slim = tf.contrib.slim
from nets import vgg
from preprocessing import vgg_preprocessing

# Load the mean pixel values and the function
# that performs the subtraction from each pixel
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN


def extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping):

    vgg_16_keys = vgg_16_variables_mapping.keys()

    vgg_16_without_fc8_keys = []

    for key in vgg_16_keys:

        if 'fc8' not in key:
            vgg_16_without_fc8_keys.append(key)

    result = {key: vgg_16_variables_mapping[key] for key in vgg_16_without_fc8_keys}
    
    return result



def FCN_32s(image_batch_tensor,
            number_of_classes,
            is_training):
    
    with tf.variable_scope("fcn_32s") as fcn_32s_scope:

        upsample_factor = 32

        # Convert image to float32 before subtracting the
        # mean pixel value
        image_batch_float = tf.to_float(image_batch_tensor, name='ToFloat')

        # Subtract the mean pixel value from each pixel
        mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

        upsample_filter_np = bilinear_upsample_weights(upsample_factor,
                                                       number_of_classes)

        upsample_filter_tensor = tf.constant(upsample_filter_np)

        # Define the model that we want to use -- specify to use only two classes at the last layer
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
        upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor,
                                         output_shape=upsampled_logits_shape,
                                         strides=[1, upsample_factor, upsample_factor, 1])

        # Let's map the original vgg-16 variable names
        # to the variables in our model. This is done
        # to make it possible to use assign_from_checkpoint_fn()
        # while providing this mapping.
        vgg_16_variables_mapping = {}

        vgg_16_variables = slim.get_variables(fcn_32s_scope)

        for variable in vgg_16_variables:

            # Here we remove the part of a name of the variable
            # that is responsible for the current variable scope
            original_vgg_16_checkpoint_string = variable.name[len(fcn_32s_scope.original_name_scope):-2]
            vgg_16_variables_mapping[original_vgg_16_checkpoint_string] = variable

    return upsampled_logits, vgg_16_variables_mapping