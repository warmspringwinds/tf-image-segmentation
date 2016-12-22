import sys
import tensorflow as tf
from ..utils.upsampling import bilinear_upsample_weights

# Using custom slim repository
sys.path.append("/home/dpakhom1/workspace/my_models/slim/")
slim = tf.contrib.slim
from nets import vgg
from preprocessing import vgg_preprocessing

def FCN_32s(image_tensor, number_of_classes):
    
    upsample_factor = 32
    
    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image_tensor, name='ToFloat')

    # Subtract the mean pixel value from each pixel
    # TODO: remove the _mean_image_subtraction function
    # dependency
    mean_centered_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])

    processed_images = tf.expand_dims(mean_centered_image, 0)

    upsample_filter_np = bilinear_upsample_weights(upsample_factor,
                                                   number_of_classes)

    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Define the model that we want to use -- specify to use only two classes at the last layer
    # TODO: make pull request to get this custom vgg feature accepted
    # to avoid using custom slim repo.
    with slim.arg_scope(vgg.vgg_arg_scope()):

        logits, end_points = vgg.vgg_16(processed_images,
                               num_classes=number_of_classes,
                               is_training=is_training_placeholder,
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
    
    return upsampled_logits