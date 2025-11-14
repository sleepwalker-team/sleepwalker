from synthesis.UTime_Repo import UtimeRepository

from cosy.types import Constructor, Literal
from cosy.synthesizer import Synthesizer

import torch

"""
This will be a simple comment based guide to specify a synthesis request.

We start with the dataset. Currently only the ABC dataset is supported.
The ABC dataset requires the following parameters to be specified:
- annotator: The annotator to use. Currently only "nsrr" and "profusion" are supported.
- channels: The channels to use. This should be a tuple of channel names.
- num_workers: The number of workers to use for data loading.
- sample_frequency: The sample frequency to use. This should be an integer.
- event_mapping: The event mapping to use. This should be a dict represented as a sequencee (tuple) of tuples, 
                 because parameters must be hashable and dicts aren't.
- online_filtering: Whether to use online filtering. This should be a boolean.
- total_input: The total input length. This should be a string, e.g. "30s".
- target_resolution: The target resolution. This should be a string, e.g. "1s".

With our synthesis you have the choice to fix a parameter or introduce variance over a substitution space.
Substitution spaces always need to be specified and fixed elements need to be elements of the substitution space.
Otherwise the synthesis result will be empty.

Also we need to specify the edf_folder path for the dataset.
Let's do this per parameter for the ABC dataset:
"""
edf_folder = "/Users/felixlaarmann/Desktop/Projekte/abc/polysomnography"

# annotator substitution are already fixed in the repo as only "nsrr" and "profusion" are supported
annotator_choice = "nsrr"  #  set this value to None, if it should be varied over the substitution space

# channels
channel_substitutions = [("Sp02", "ECG1", "ECG2", "Thor"), ] # substitution space for channels
channel_choice = ("Sp02", "ECG1", "ECG2", "Thor")  # set this value to None, if it should be varied over the substitution space

# num_workers
num_workers_substitutions = [4, 8, 16]  # substitution space for num_workers
num_workers_choice = 8  # set this value to None, if it should be varied over the substitution space

# sample_frequency
sample_frequency_substitutions = [10, 50, 100]  # substitution space for sample_frequency
sample_frequency_choice = 100  # set this value to None, if it should be varied over the substitution space

# event_mapping
event_mapping_substitutions = [(
    ("hypopnea|hypopnea", "hypopnea"),
    ("central apnea|central apnea", "apnea"),
    ("obstructive apnea|obstructive apnea", "apnea"),),
]  # substitution space for event_mapping
event_mapping_choice = (
    ("hypopnea|hypopnea", "hypopnea"),
    ("central apnea|central apnea", "apnea"),
    ("obstructive apnea|obstructive apnea", "apnea"),)  # set this value to None, if it should be varied over the substitution space

# online_filtering
online_filtering_choice = True  # set this value to None, if it should be varied over the substitution space

# total_input
total_input_substitutions = ["15s", "30s", "60s"]  # substitution space for total_input
total_input_choice = "30s"  # set this value to None, if it should be varied over the substitution space

# target_resolution
target_resolution_substitutions = ["1s", "2s"]  # substitution space for target_resolution
target_resolution_choice = "1s"  # set this value to None, if it should be varied over the substitution space

"""
Now that we have specified the dataset parameters, we can proceed to define the U-Time model architecture 
and training parameters.

A U-Time model is modeled as a sequence of encode, decoder pairs with a bottleneck "at the bottom".
All encoder, decoder pairs have the same (homogeneous) parameters, except for the first pair, which
can have different parameters.
The output of the U-Time model is passed to a final convolutional layer and a linear classifier.
By choosing a loss function and a sequence of preprocessors the Utime model is fully specified.

Lets go through the substitution spaces and choices for each parameter:
"""

# Each encoder, decoder pair consists of a small convolutional neural network with the following parameters:
# - convolution: The type of convolution to use. "simple_convolution" and "depthwise_separable_convolution" are
# currently supported.
# Since the first layer may differ, you need to specify the choices for the first layer and
# the homogeneous rest separately.

first_layer_convolution_type_choice = "simple_convolution"
convolution_type_choice = "simple_convolution"
# set this value to None, if it should be varied over the substitution space

# - convolution parameters: stride, padding, dilation
convolution_stride_substitutions = [1, 2]  # substitution space for convolution stride
first_layer_convolution_stride_choice = 1  # set this value to None, if it should be varied over the substitution space
convolution_stride_choice = 1  # set this value to None, if it should be varied over the substitution space

convolution_padding_substitutions = [0, 1, 2]  # substitution space for convolution padding
first_layer_convolution_padding_choice = 0  # set this value to None, if it should be varied over the substitution space
convolution_padding_choice = 0  # set this value to None, if it should be varied over the substitution space

convolution_dilation_substitutions = [1, 2]  # substitution space for convolution dilation
first_layer_convolution_dilation_choice = 1  # set this value to None, if it should be varied over the substitution space
convolution_dilation_choice = 1  # set this value to None, if it should be varied over the substitution space

# the length of the sequence of encoder, decoder pairs is defined by providing a list of input channels for each pair.
# channel outputs will be computed from the input channels and the fact, that the output of the u-structure is equal to
# its input.
# But first we need to define the substitution space for numbers of channels
channel_input_substitutions = [1, 2, 4, 8, 64, 128, 256, 512]  # substitution space for number of channels
channel_inputs_choice = (4, 512, 256) # Since the length needs to be determined, only the values within the sequence
                                      # can be varied over the substitution space by setting them to None.
                                      # E.g. (None, 512, None)
                                      # The first (most left) value will be the input channels of the first layer

# convolution kernel sizes
convolution_kernel_size_substitutions = [1, 2, 3, 5]  # substitution space for convolution kernel sizes
convolution_kernel_sizes_choice = (2, 3, 5) # Since the length needs to be determined, only the values within the sequence
                                      # can be varied over the substitution space by setting them to None.
                                      # E.g. (None, 3, None)
                                      # The first (most left) value will be the convolution kernel size of the first layer


# maxpool parameters: kernel_size, stride, padding, dilation
maxpool_size_substitutions = [3, 5]  # substitution space for maxpool kernel size
maxpool_size_choice = (5,3,5) # Since the length needs to be determined, only the values within the sequence
                                      # can be varied over the substitution space by setting them to None.
                                      # E.g. (None, 3, None)
                                      # The first (most left) value will be the maxpool kernel size of the first layer

maxpool_stride_substitutions = [1, ]  # substitution space for maxpool stride
first_layer_maxpool_stride_choice = 1  # set this value to None, if it should be varied over the substitution space
maxpool_stride_choice = 1  # set this value to None, if it should be varied over the substitution space

maxpool_padding_substitutions = [0, ]  # substitution space for maxpool padding
first_layer_maxpool_padding_choice = 0  # set this value to None, if it should be varied over the substitution space
maxpool_padding_choice = 0  # set this value to None, if it should be varied over the substitution space

maxpool_dilation_substitutions = [1, ]  # substitution space for maxpool dilation
first_layer_maxpool_dilation_choice = 1  # set this value to None, if it should be varied over the substitution space
maxpool_dilation_choice = 1  # set this value to None, if it should be varied over the substitution space

# - bias: can only be True or False
first_layer_convolution_bias_choice = True  # set this value to None, if it should be varied over the substitution space
convolution_bias_choice = True  # set this value to None, if it should be varied over the substitution space

# - activation: currently only "ReLu", "ELU" and "Tanh" are supported
first_layer_activation_choice = "ReLu"  # set this value to None, if it should be varied over the substitution space
activation_choice = "ReLu"  # set this value to None, if it should be varied over the substitution space

# - dropout_p: dropout probability
dropout_p_substitutions = [0.1, 0.2, 0.3]  # substitution space for dropout probability
first_layer_dropout_p_choice = 0.1  # set this value to None, if it should be varied over the substitution space
dropout_p_choice = 0.1  # set this value to None, if it should be varied over the substitution space

# normalization: currently only "batch_norm", "conv1d_layer_norm" and "channel_wise_norm", are supported
first_layer_normalization_choice = "channel_wise_norm"  # set this value to None, if it should be varied over the substitution space
normalization_choice = "channel_wise_norm"  # set this value to None, if it should be varied over the substitution space

# normalization_epsilon: hyperparameter for normalization layers
normalization_epsilon_substitutions = [1e-3, 1e-4]  # substitution space for normalization epsilon
first_layer_normalization_epsilon_choice = 1e-3  # set this value to None, if it should be varied over the substitution space
normalization_epsilon_choice = 1e-3  # set this value to None, if it should be varied over the substitution space

# the bottleneck shares its parameters with the homogeneous layers, and is therefore only parametrized by:
# - in_and_out: number of input and output channels is equal
bottleneck_in_and_out_choice = 64  # set this value to None, if it should be varied over the previously defined
                                   # substitution space of valid channel numbers
# - kernel_size: size of the convolutional kernel
bottleneck_kernel_size_choice = 1  # set this value to None, if it should be varied over the previously defined
                                   # substitution space of valid convolution kernel sizes

# final convolutional layer parameters: The previously defined substitution spaces for convolutional layers apply here
# as well.
# - convolution: The type of convolution to use. "simple_convolution" and "depthwise_separable_convolution" are
# currently supported.
final_convolution_type_choice = "simple_convolution"
# set this value to None, if it should be varied over the substitution space
# - convolution kernel size
final_convolution_kernel_size_choice = 1  # set this value to None, if it should be varied over the substitution space
# - convolution parameters: stride, padding, dilation
final_convolution_stride_choice = 1  # set this value to None, if it should be varied over the substitution space
final_convolution_padding_choice = 0  # set this value to None, if it should be varied over the substitution space
final_convolution_dilation_choice = 1  # set this value to None, if it should be varied over the substitution space
final_convolution_bias_choice = True  # set this value to None, if it should be varied over the substitution space

# - linear classifier parameters: input, output, bias
# I was lazy and currently valid input and output dimensions come from the substitution space for channel numbers.
# This will be changed in the future to be more flexible.
linear_classifier_input_choice = 512  # set this value to None , if it should be varied over the substitution space
linear_classifier_output_choice = 2  # set this value to None , if it should be varied over the substitution space
linear_classifier_bias_choice = False  # set this value to None , if it should be varied over the substitution space

# - loss function: currently only "BCE_with_logits", "CrossEntropy", "MAE" and "MSE" are supported
loss_function_choice = "CrossEntropy"  # set this value to None , if it should be varied over the substitution space

"""
- preprocessors: currently supported are "ChannelSampler", "FIR", "EmpiricalClipScaler", "RobustScaler", "Spectogram", 
                  "Crop", "Normalize" and "ZNormalize"
each preprocessor is modelled as a tuple, where the first element is the name of the preprocessor and the following 
elements are the parameters of the preprocessor.

("ChannelSampler", n)
or
("Crop", total_input, sampling_rate, where)
or
("EmpiricalClipScaler", q, scale)
or
("FIR", sampling_rate, channels, filter_params, zero_phase)
or
("Normalize",)
or
("RobustScaler", lower_quantile, upper_quantile)
or
("Spectogram", n_fft, hop_length, win_length, epoch_len_samples)
or
("ZNormalize", use_global_statistics)
"""

# First we need to provide substitution spaces for all possible parameters of the preprocessors.
# Even if we don't fix them within synthesis.

preprocessor_channel_sampler_n_substitutions = [4]
preprocessor_crop_total_input_substitutions = [128]
preprocessor_crop_sampling_rate_substitutions = [64]
preprocessor_empirical_clip_scaler_q_substitutions = [0.9]
preprocessor_empirical_clip_scaler_scale_substitutions = [1]
preprocessor_fir_sampling_rate_substitutions = [64]
preprocessor_fir_channels_substitutions = ["channel"]
preprocessor_fir_filter_params_substitutions = [(("channel", "filter_param"),)]
preprocessor_robust_scaler_lower_quantile_substitutions = [0.25]
preprocessor_robust_scaler_upper_quantile_substitutions = [0.75]
preprocessor_spectogram_n_fft_substitutions = [256]
preprocessor_spectogram_hop_length_substitutions = [64]
preprocessor_spectogram_win_length_csubstitutions = [torch.hamming_window(256)]
preprocessor_spectogram_epoch_len_samples_substitutions = [1]

# Now we can choose the actual preprocessor sequence for our synthesis request.
preprocessor_sequence_choice = (
    ("ChannelSampler", 4),
    # ("FIR", 64, "channel", (("channel", "filter_param"),), False),
    # None,
    #("RobustScaler", None, None)
) # a whole preprocessor tuple can be set to None, if it should be varied over the substitution space of preprocessors,
# or individual parameters within a preprocessor tuple can be set to None, if they should be varied over their
# respective substitution space

"""
Now we just need to specify training parameters:
- Sampler: currently only "RandomSampler" or "NoSampler" (which corresponds to sample=None in sleepwalker) are supported.
- n_samples: number of samples to draw from the dataset for training
- replacement: whether to sample with replacement (only for RandomSampler)

- batch_size: batch size for training
- epochs: number of epochs to train
- optimizer: optimizer is a tuple similar to preprocessors, where the first element is the optimizer name and the following
             elements are the optimizer parameters. The choices are as follows:

            ("Adagrad", learning_rate, learning_rate_decay, weight_decay, initial_accumulator_value, eps)
            ("Adam", learning_rate, betas, eps, weight_decay, amsgrad)
            ("AdamW", learning_rate, betas, eps, weight_decay, amsgrad)
            ("Adamax", learning_rate, betas, eps, weight_decay)
            ("SGD", learning_rate, momentum, dampening, weight_decay, nesterov)
            
- lr_scheduler: learning rate scheduler is also a tuple, where the first element is the scheduler name and the following
                elements are the scheduler parameters. The choices are as follows:
                
            ("LinearLR", start_factor, end_factor, total_iters, last_epoch)
            ("StepLR", step_size, gamma, last_epoch)
            ("ExponentialLR", gamma, last_epoch)
"""
sampler_choice = "RandomSampler"  # set this value to None, if it should be varied over the substitution space
n_samples_substitutions = [10]  # substitution space for n_samples
n_samples_choice = 10  # set this value to None, if it should be varied over the substitution space
replacement_choice = True  # set this value to None, if it should be varied over the substitution space

batch_size_choice = 8  # Sebastian told me that variance in batch size doesn't make sense, so we fix it here.

epochs_choice = 5 # This value must be set and is treated as a global constant, since variance in epochs doesn't
                  # make sense.

# Again we need to provide substitution spaces for all possible parameters of the optimizers and lr_schedulers.
# Even if we don't fix them within synthesis.

optimizer_learning_rate_substitutions = [1e-3]
optimizer_learning_rate_decay_substitutions = [0]
optimizer_weight_decay_substitutions = [0, 1e-4]
optimizer_eps_substitutions = [1e-10]
optimizer_beta_substitutions = [(0.9, 0.999)]
optimizer_initial_accumulator_value_substitutions = [0]
optimizer_momentum_substitutions = [0]
optimizer_dampening_substitutions = [0]
lr_scheduler_start_factor_substitutions = [1]
lr_scheduler_end_factor_substitutions = [1e-2]
lr_scheduler_total_iters_substitutions = [50]
lr_scheduler_step_size_substitutions = [30]
lr_scheduler_gamma_substitutions = [0.1, 0.95]
lr_scheduler_last_epoch_substitutions = [-1]

"""
Just for simplicity, the possible optimzer structures are listed here again:

("Adagrad", learning_rate, learning_rate_decay, weight_decay, initial_accumulator_value, eps)
("Adam", learning_rate, betas, eps, weight_decay, amsgrad)
("AdamW", learning_rate, betas, eps, weight_decay, amsgrad)
("Adamax", learning_rate, betas, eps, weight_decay)
("SGD", learning_rate, momentum, dampening, weight_decay, nesterov)
"""

optimizer_choice = ("Adam", 1e-3, (0.9, 0.999), 1e-10, 0, True)  # set this value to None, if it should be varied over
# the substitution space of all optimizer and all parameters for each optimizer
# (yes, this can be a huge cartesian product).
# optimizer parameters can be None to be varied over the substitution space, similar to preprocessors.

"""
Just for simplicity, the possible lr_scheduler structures are listed here again:
("LinearLR", start_factor, end_factor, total_iters, last_epoch)
("StepLR", step_size, gamma, last_epoch)
("ExponentialLR", gamma, last_epoch)
"""

lr_scheduler_choice = ("LinearLR", 1, 1e-2, 50, -1)  # set this value to None, if it should be varied over
# the substitution space of all lr_schedulers and all parameters for each lr_scheduler
# (yes, this can be a huge cartesian product).
# lr_scheduler parameters can be None to be varied over the substitution space, similar to preprocessors.

"""
Okidoki, now everything is specified.
Last, but not least, you need to make two more choices:
Which algebra to use to interpret the synthesized terms and up to how many synthesis results you want to enumerate.
If there are less results, enumeration will stop earlier.

You may choose between:
- "pretty_term_algebra": This will print the synthesized term in a human readable format. 
                              Mostly useful for debugging the repository.
- "python_code_algebra": This will generate python code for the synthesized term and write it to a file in the 
                              synthesis_output folder.
- "torch_algebra": This is a objective function for bayesian optimization, that directly interprets the synthesized 
                        term as a pytorch model and trains it on the specified dataset with the specified training 
                        parameters. Basically it's semantic is equivalent to the python code algebra, but it doesn't 
                        generate python code files and is interpreted directly at runtime.                            
"""
number_of_synthesis_results = 10
interpretation = "python_code_algebra"

"""
Everythings set up now, so simply run this script to start the synthesis process!

Some remarks:
If you used no None for variance and everything else is provided correctly, you should have described exactly one
system and therefore get exactly one synthesis result.

TODO: When tidying up and restructuring the branch, I introduced a bug that leads to multiple synthesis results even if 
      no variance is introduced. I will fix this soon.

If you used None for some parameters, the synthesis process will explore the substitution space and return
all valid combinations of parameters that lead to a valid system within the provided substitution spaces.
This may lead to a huge number of synthesis results, depending on how many parameters you set to None and how
large the respective substitution spaces are.

If you made mistakes in the specification, you get zero synthesis results.
"""

if __name__ == "__main__":

    repo = UtimeRepository(dimension_choices=channel_input_substitutions,
                           normalization_eps_choices=normalization_epsilon_substitutions,
                           dropout_p_choices=dropout_p_substitutions,
                           convolution_kernel_size_choices=convolution_kernel_size_substitutions,
                           convolution_stride_choices=convolution_stride_substitutions,
                           convolution_padding_choices=convolution_padding_substitutions,
                           convolution_dilations_choices=convolution_dilation_substitutions,
                           maxpool_size_choices=maxpool_size_substitutions,
                           maxpool_stride_choices=maxpool_stride_substitutions,
                           maxpool_padding_choices=maxpool_padding_substitutions,
                           maxpool_dilation_choices=maxpool_dilation_substitutions,
                           preprocessor_channel_sampler_n_choices=preprocessor_channel_sampler_n_substitutions,
                           preprocessor_crop_total_input_choices=preprocessor_crop_total_input_substitutions,
                           preprocessor_crop_sampling_rate_choices=preprocessor_crop_sampling_rate_substitutions,
                           preprocessor_empirical_clip_scaler_q_choices=preprocessor_empirical_clip_scaler_q_substitutions,
                           preprocessor_empirical_clip_scaler_scale_choices=preprocessor_empirical_clip_scaler_scale_substitutions,
                           preprocessor_fir_sampling_rate_choices=preprocessor_fir_sampling_rate_substitutions,
                           preprocessor_fir_channels_choices=preprocessor_fir_channels_substitutions,
                           preprocessor_fir_filter_params_choices=preprocessor_fir_filter_params_substitutions,
                           preprocessor_robust_scaler_lower_quantile_choices=preprocessor_robust_scaler_lower_quantile_substitutions,
                           preprocessor_robust_scaler_upper_quantile_choices=preprocessor_robust_scaler_upper_quantile_substitutions,
                           preprocessor_spectogram_n_fft_choices=preprocessor_spectogram_n_fft_substitutions,
                           preprocessor_spectogram_hop_length_choices=preprocessor_spectogram_hop_length_substitutions,
                           preprocessor_spectogram_win_length_choices=preprocessor_spectogram_win_length_csubstitutions,
                           preprocessor_spectogram_epoch_len_samples_choices=preprocessor_spectogram_epoch_len_samples_substitutions,
                           n_samples=n_samples_substitutions,
                           abc_channel_choices=channel_substitutions,
                           abc_event_mapping=event_mapping_substitutions,
                           abc_num_workers=num_workers_substitutions,
                           abc_sample_frequency=sample_frequency_substitutions,
                           abc_total_input=total_input_substitutions,
                           abc_target_resolution=target_resolution_substitutions,
                           batch_size=[batch_size_choice],
                           epochs=epochs_choice,
                           optimizer_learning_rate=optimizer_learning_rate_substitutions,
                           optimizer_learning_rate_decay=optimizer_learning_rate_decay_substitutions,
                           optimizer_weight_decay=optimizer_weight_decay_substitutions,
                           optimizer_eps=optimizer_eps_substitutions,
                           optimizer_beta=optimizer_beta_substitutions,
                           optimizer_initial_accumulator_value=optimizer_initial_accumulator_value_substitutions,
                           optimizer_momentum=optimizer_momentum_substitutions,
                           optimizer_dampening=optimizer_dampening_substitutions,
                           lr_scheduler_start_factor_choices=lr_scheduler_start_factor_substitutions,
                           lr_scheduler_end_factor_choices=lr_scheduler_end_factor_substitutions,
                           lr_scheduler_total_iters_choices=lr_scheduler_total_iters_substitutions,
                           lr_scheduler_step_size_choices=lr_scheduler_step_size_substitutions,
                           lr_scheduler_gamma_choices=lr_scheduler_gamma_substitutions,
                           lr_scheduler_last_epoch_choices=lr_scheduler_last_epoch_substitutions,
                           edf_path=edf_folder)

    target = Constructor("trainer",
                     Constructor("u_model",
                                 Constructor("u_classifier",
                                             Constructor("dimensions", Literal(channel_inputs_choice))
                                             & Constructor("kernel_sizes", Literal(convolution_kernel_sizes_choice))
                                             & Constructor("maxpool_sizes", Literal(maxpool_size_choice))
                                             )
                                 & Constructor("u_first_level",
                                               Constructor("convolution", Literal(first_layer_convolution_type_choice))
                                               & Constructor("convolution_stride", Literal(first_layer_convolution_stride_choice))
                                               & Constructor("convolution_padding", Literal(first_layer_convolution_padding_choice))
                                               & Constructor("convolution_dilation", Literal(first_layer_convolution_dilation_choice))
                                               & Constructor("bias", Literal(first_layer_convolution_bias_choice))
                                               & Constructor("activation", Literal(first_layer_activation_choice))
                                               & Constructor("dropout_p", Literal(first_layer_dropout_p_choice))
                                               & Constructor("normalization", Literal(first_layer_normalization_choice))
                                               & Constructor("normalization_epsilon", Literal(first_layer_normalization_epsilon_choice))
                                               & Constructor("maxpool_stride", Literal(first_layer_maxpool_stride_choice))
                                               & Constructor("maxpool_padding", Literal(first_layer_maxpool_padding_choice))
                                               & Constructor("maxpool_dilation", Literal(first_layer_maxpool_dilation_choice))
                                               )
                                 & Constructor("bottleneck",
                                               Constructor("in_and_out", Literal(bottleneck_in_and_out_choice))
                                               & Constructor("kernel_size", Literal(bottleneck_kernel_size_choice))
                                               )
                                 & Constructor("homogeneous",
                                               Constructor("convolution", Literal(convolution_type_choice))
                                               & Constructor("convolution_stride", Literal(convolution_stride_choice))
                                               & Constructor("convolution_padding", Literal(convolution_padding_choice))
                                               & Constructor("convolution_dilation", Literal(convolution_dilation_choice))
                                               & Constructor("bias", Literal(convolution_bias_choice))
                                               & Constructor("activation", Literal(activation_choice))
                                               & Constructor("dropout_p", Literal(dropout_p_choice))
                                               & Constructor("normalization", Literal(normalization_choice))
                                               & Constructor("normalization_epsilon", Literal(normalization_epsilon_choice))
                                               & Constructor("maxpool_stride", Literal(maxpool_stride_choice))
                                               & Constructor("maxpool_padding", Literal(maxpool_padding_choice))
                                               & Constructor("maxpool_dilation", Literal(maxpool_dilation_choice))
                                               )
                                 & Constructor("u_final_conv",
                                               Constructor("kernel_size", Literal(final_convolution_kernel_size_choice))
                                               & Constructor("convolution", Literal(final_convolution_type_choice))
                                               & Constructor("convolution_stride", Literal(final_convolution_stride_choice))
                                               & Constructor("convolution_padding", Literal(final_convolution_padding_choice))
                                               & Constructor("convolution_dilation", Literal(final_convolution_dilation_choice))
                                               & Constructor("bias", Literal(final_convolution_bias_choice))
                                               )
                                 & Constructor("u_linear_classifier",
                                               Constructor("linear_layer",
                                                           Constructor("input", Literal(linear_classifier_input_choice))
                                                           & Constructor("output", Literal(linear_classifier_output_choice))
                                                           & Constructor("bias", Literal(linear_classifier_bias_choice))
                                                           )
                                               )
                                 & Constructor("loss_function", Literal(loss_function_choice))
                                 & Constructor("preprocessors", Literal(preprocessor_sequence_choice))
                                 )
                     & Constructor("dataloader",
                                   Constructor("Sampler",
                                               Literal(sampler_choice)
                                               & Constructor("replacement", Literal(replacement_choice))
                                               & Constructor("num_samples", Literal(n_samples_choice))
                                               )
                                   & Constructor("Dataset",
                                                 Literal("ABC_Dataset")
                                                 & Constructor("annotator", Literal(annotator_choice))
                                                 & Constructor("channels", Literal(channel_choice))
                                                 & Constructor("num_workers", Literal(num_workers_choice))
                                                 & Constructor("sample_frequency", Literal(sample_frequency_choice))
                                                 & Constructor("event_mapping", Literal(event_mapping_choice))
                                                 & Constructor("online_filtering", Literal(online_filtering_choice))
                                                 & Constructor("total_input", Literal(total_input_choice))
                                                 & Constructor("target_resolution", Literal(target_resolution_choice))
                                                 )
                                   & Constructor("batch_size", Literal(batch_size_choice))
                                   )
                     & Constructor("optimizer", Literal(optimizer_choice))
                     & Constructor("lr_scheduler", Literal(lr_scheduler_choice))
                     )


    synthesizer = Synthesizer(repo.specification(), {})

    search_space = synthesizer.construct_solution_space(target).prune()

    trees = search_space.enumerate_trees(target, number_of_synthesis_results)

    for i, t in enumerate(trees):
        if interpretation == "pretty_term_algebra":
            print(t.interpret(repo.pretty_term_algebra()))
        elif interpretation == "python_code_algebra":
            with open(f"./synthesis_output/train_utime_{i}.py", "w") as f:
                f.write(t.interpret(repo.python_code_algebra()))
        elif interpretation == "pytorch_algebra":
            t.interpret(repo.torch_algebra())
        else:
            raise ValueError(f"Unknown interpretation: {interpretation}")
