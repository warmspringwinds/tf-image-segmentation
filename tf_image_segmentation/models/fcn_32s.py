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

def FCN_32s(image_batch_tensor,
            number_of_classes,
            is_training,
            vgg_16_checkpoint_filename):
    
    with tf.variable_scope("fcn_32s") as fcn_32_scope:
    
        upsample_factor = 32

        # Convert image to float32 before subtracting the
        # mean pixel value
        image_batch_float = tf.to_float(image_batch_tensor, name='ToFloat')

        # Subtract the mean pixel value from each pixel
        mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

        # processed_images = tf.expand_dims(mean_centered_image, 0)

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
        
        vgg_fc8_vars_full_scope_name = fcn_32_scope.original_name_scope + 'vgg_16/fc8'
        vgg_vars_full_scope_name = fcn_32_scope.original_name_scope + 'vgg_16'

        vgg_vars_only_fc8 = slim.get_variables(scope=vgg_fc8_vars_full_scope_name)
        vgg_vars_without_fc8 = slim.get_variables_to_restore(include=[vgg_vars_full_scope_name],
                                                        exclude=[vgg_fc8_vars_full_scope_name])
        
        recover_mapping = {}

        for var in vgg_vars_without_fc8:

            original_string = var.name[len(fcn_32_scope.original_name_scope):-2]
            recover_mapping[original_string] = var
        
        # Create an OP that performs the initialization of
        # values of variables to the values from VGG.
        read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
                                           vgg_16_checkpoint_filename,
                                           recover_mapping)

        # Initializer for new fc8 weights -- for two classes.
        vgg_fc8_weights_initializer = tf.variables_initializer(vgg_vars_only_fc8)

        def fcn_32s_initialization_func(current_session):

            read_vgg_weights_except_fc8_func(current_session)
            current_session.run(vgg_fc8_weights_initializer)

        return upsampled_logits, fcn_32s_initialization_func