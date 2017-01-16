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



def FCN_16s(image_batch_tensor,
            number_of_classes,
            is_training):

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_batch_float = tf.to_float(image_batch_tensor, name='ToFloat')

    # Subtract the mean pixel value from each pixel
    mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

    upsample_filter_factor_2_np = bilinear_upsample_weights(factor=2,
                                                            number_of_classes=number_of_classes)

    upsample_filter_factor_16_np = bilinear_upsample_weights(factor=16,
                                                             number_of_classes=number_of_classes)

    upsample_filter_factor_2_tensor = tf.constant(upsample_filter_factor_2_np)
    upsample_filter_factor_16_tensor = tf.constant(upsample_filter_factor_16_np)

    with tf.variable_scope("fcn_16s")  as fcn_16s_scope:
        # Define the model that we want to use -- specify to use only two classes at the last layer
        # TODO: make pull request to get this custom vgg feature accepted
        # to avoid using custom slim repo.
        with slim.arg_scope(vgg.vgg_arg_scope()):

            ## Original FCN-32s model definition

            last_layer_logits, end_points = vgg.vgg_16(mean_centered_image_batch,
                                                       num_classes=number_of_classes,
                                                       is_training=is_training,
                                                       spatial_squeeze=False,
                                                       fc_conv_padding='SAME')


            last_layer_logits_shape = tf.shape(last_layer_logits)


            # Calculate the ouput size of the upsampled tensor
            last_layer_upsampled_by_factor_2_logits_shape = tf.pack([
                                                                  last_layer_logits_shape[0],
                                                                  last_layer_logits_shape[1] * 2,
                                                                  last_layer_logits_shape[2] * 2,
                                                                  last_layer_logits_shape[3]
                                                                 ])

            # Perform the upsampling
            last_layer_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(last_layer_logits,
                                                                     upsample_filter_factor_2_tensor,
                                                                     output_shape=last_layer_upsampled_by_factor_2_logits_shape,
                                                                     strides=[1, 2, 2, 1])

            ## Adding the skip here for FCN-16s model

            pool4_features = end_points['fcn_16s/vgg_16/pool4']

            # We zero initialize the weights to start training with the same
            # accuracy that we ended training FCN-32s

            pool4_logits = slim.conv2d(pool4_features,
                                       number_of_classes,
                                       [1, 1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='pool4_fc')

            fused_last_layer_and_pool4_logits = pool4_logits + last_layer_upsampled_by_factor_2_logits

            fused_last_layer_and_pool4_logits_shape = tf.shape(fused_last_layer_and_pool4_logits)


            # Calculate the ouput size of the upsampled tensor
            fused_last_layer_and_pool4_upsampled_by_factor_16_logits_shape = tf.pack([
                                                                          fused_last_layer_and_pool4_logits_shape[0],
                                                                          fused_last_layer_and_pool4_logits_shape[1] * 16,
                                                                          fused_last_layer_and_pool4_logits_shape[2] * 16,
                                                                          fused_last_layer_and_pool4_logits_shape[3]
                                                                         ])

            # Perform the upsampling
            fused_last_layer_and_pool4_upsampled_by_factor_16_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits,
                                                                        upsample_filter_factor_16_tensor,
                                                                        output_shape=fused_last_layer_and_pool4_upsampled_by_factor_16_logits_shape,
                                                                        strides=[1, 16, 16, 1])

            fcn_32s_variables_mapping = {}

            fcn_16s_variables = slim.get_variables(fcn_16s_scope)

            for variable in fcn_16s_variables:
                
                # We only need FCN-32s variables to resture from checkpoint
                # Variables of FCN-16s should be initialized
                if 'pool4_fc' in variable.name:
                    continue

                # Here we remove the part of a name of the variable
                # that is responsible for the current variable scope
                original_fcn_32s_checkpoint_string = 'fcn_32s/' +  variable.name[len(fcn_16s_scope.original_name_scope):-2]
                fcn_32s_variables_mapping[original_fcn_32s_checkpoint_string] = variable

    return fused_last_layer_and_pool4_upsampled_by_factor_16_logits, fcn_32s_variables_mapping