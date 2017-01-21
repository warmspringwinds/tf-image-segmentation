from nets import vgg
import tensorflow as tf
from preprocessing import vgg_preprocessing
from ..utils.upsampling import bilinear_upsample_weights

slim = tf.contrib.slim

# Mean values for VGG-16
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN



def FCN_8s(image_batch_tensor,
           number_of_classes,
           is_training):
    """Returns the FCN-8s model definition.
    The function returns the model definition of a network that was described
    in 'Fully Convolutional Networks for Semantic Segmentation' by Long et al.
    The network subsamples the input by a factor of 32 and uses three bilinear
    upsampling layers to upsample prediction by a factor of 32. This means that
    if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use 
    adapt_network_for_any_size_input(FCN_8s, 32). Note: the upsampling kernel
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
    fcn_16s_variables_mapping : dict {string: variable}
        Dict which maps the FCN-8s model's variables to FCN-16s checkpoint variables
        names. We need this to initilize the weights of FCN-8s model with FCN-16s from
        checkpoint file. Look at ipython notebook for examples.
    """

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_batch_float = tf.to_float(image_batch_tensor)

    # Subtract the mean pixel value from each pixel
    mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

    upsample_filter_factor_2_np = bilinear_upsample_weights(factor=2,
                                                            number_of_classes=number_of_classes)

    upsample_filter_factor_8_np = bilinear_upsample_weights(factor=8,
                                                             number_of_classes=number_of_classes)

    upsample_filter_factor_2_tensor = tf.constant(upsample_filter_factor_2_np)
    upsample_filter_factor_8_tensor = tf.constant(upsample_filter_factor_8_np)

    with tf.variable_scope("fcn_8s")  as fcn_8s_scope:
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
            
            # We created vgg in the fcn_8s name scope -- so
            # all the vgg endpoints now are prepended with fcn_8s name
            pool4_features = end_points['fcn_8s/vgg_16/pool4']

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
            fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape = tf.pack([
                                                                          fused_last_layer_and_pool4_logits_shape[0],
                                                                          fused_last_layer_and_pool4_logits_shape[1] * 2,
                                                                          fused_last_layer_and_pool4_logits_shape[2] * 2,
                                                                          fused_last_layer_and_pool4_logits_shape[3]
                                                                         ])

            # Perform the upsampling
            fused_last_layer_and_pool4_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits,
                                                                        upsample_filter_factor_2_tensor,
                                                                        output_shape=fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape,
                                                                        strides=[1, 2, 2, 1])
            
            
            ## Adding the skip here for FCN-8s model

            pool3_features = end_points['fcn_8s/vgg_16/pool3']
            
            # We zero initialize the weights to start training with the same
            # accuracy that we ended training FCN-32s

            pool3_logits = slim.conv2d(pool3_features,
                                       number_of_classes,
                                       [1, 1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='pool3_fc')
            
            
            fused_last_layer_and_pool4_logits_and_pool_3_logits = pool3_logits + \
                                            fused_last_layer_and_pool4_upsampled_by_factor_2_logits
            
            
            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape = tf.shape(fused_last_layer_and_pool4_logits_and_pool_3_logits)
            
            
            # Calculate the ouput size of the upsampled tensor
            fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape = tf.pack([
                                                                          fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[0],
                                                                          fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[1] * 8,
                                                                          fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[2] * 8,
                                                                          fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[3]
                                                                         ])

            # Perform the upsampling
            fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits_and_pool_3_logits,
                                                                        upsample_filter_factor_8_tensor,
                                                                        output_shape=fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape,
                                                                        strides=[1, 8, 8, 1])
            
            
            
            
            fcn_16s_variables_mapping = {}

            fcn_8s_variables = slim.get_variables(fcn_8s_scope)

            for variable in fcn_8s_variables:
                
                # We only need FCN-16s variables to resture from checkpoint
                # Variables of FCN-8s should be initialized
                if 'pool3_fc' in variable.name:
                    continue

                # Here we remove the part of a name of the variable
                # that is responsible for the current variable scope
                original_fcn_16s_checkpoint_string = 'fcn_16s/' +  variable.name[len(fcn_8s_scope.original_name_scope):-2]
                fcn_16s_variables_mapping[original_fcn_16s_checkpoint_string] = variable

    return fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits, fcn_16s_variables_mapping