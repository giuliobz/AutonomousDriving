"""CNN - LSTM config system
This file specifies deafoult congig options for the model used.
You should not change this file: is recommended copy and paste in other config.py file the own configuration and replace it in this folder.

"""

from utils.collections import AttrDict

__C = AttrDict()

#Users can get the configuration by:
#   from config.config import cfg
cfg = __C

#####################################################################################################################################
#
# DEFINE THE DIFFERENT PATH WHERE THE TREE MODEL WILL SAVE THE RESULT
#
#####################################################################################################################################
__C.SAVE_WEIGHT_PATH = { 'depth' : "/saved_models/depth/", 'multi' : "/saved_models/multi_frame/", 'single' :  "/saved_models/single_frame/"}
__C.SAVE_RESULTS_PATH = { 'depth' : "/test_results/depth_results/", 'multi' : "/test_results/multi_frame_results/", 'single' :  "/test_results/single_frame_results/"}
__C.SAVE_VIDEO_PATH = {'depth' : "/vides/depth_video/", 'multi' : "/videos/multi_frame_video/", 'single' :  "/videos/single_frame_video/"}
__C.TENSORBOARD_PATH = { 'depth' : "/tensorboard_runs/depth/", 'multi' : "/tensorboard_runs/multi_frame/", 'single' :  "/tensorboard_runs/single_frame/"}
__C.DATASET_PATH = {'modify' : "/datasets/image_dataset/Dataset/", 'real' : "/datasets/image_dataset/Dataset_real/"}
__C.CSV_DATASET_PATH = "/datasets/csv_dataset/"
#####################################################################################################################################
#
# TRAINING PARAMETERS
#
#####################################################################################################################################

__C.TRAIN = AttrDict()


#####################################################################################################################################
#
# DEFINE THE DIMENSION
#
#####################################################################################################################################

#
# Define the hidden and cell state dimension for the tree models and LSTM layers
#

__C.IN_CHANNELS = {'depth' : 4, 'multi' : 15, 'single' : 3}
__C.DIMENSION = {'depth' : 256, 'multi' : 768, 'single' : 128}
__C.LAYERS = 2


#####################################################################################################################################
#
# DEFINE THE MODELS' COMMON PARAMETERS 
#
#####################################################################################################################################

#
# Define gradient clipping, batch size, learning rate and decrement period 
# Decrement period: number of epochs after that we decrement the learning rate
# Gradient clipping: manca definizione
#  
__C.TRAIN.GRADIENT_CLIP = 5
__C.TRAIN.BATCH_SIZE = 20
__C.TRAIN.LEARNING_RATE = 0.1
__C.TRAIN.DEC_PERIOD = 20

#
# Define some information of the dataset
# Len_sequence : define the lenght of the sequence to predict
# Shuffle: indicates if the images have to be shuffle 
#
__C.TRAIN.LEN_SEQUENCES = 30
__C.TRAIN.SHUFFLE_T = 'True'
__C.TRAIN.SHUFFLE_V = 'False'


#
# Define loss function, the optimizer and the alpha and momentum optimizer parameters
# Define olso the gamma parameter for the scheduler
# SGD: 
# Alpha:
# Momentum:
# Gamma: define how much we decrease the learning rate
#
__C.TRAIN.ALPHA = 0.90
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.OPTIMIZER = 'SGD'
__C.TRAIN.LOSS = 'MSE'

#####################################################################################################################################
#
# TEST PARAMETERS
#
#####################################################################################################################################


__C.TEST = AttrDict()

#####################################################################################################################################
#
# DEFINE THE HIDDEN DIMENSION
#
#####################################################################################################################################

__C.TEST.HIDDEN_DIMENSION_DEPTH = 256
__C.TEST.HIDDEN_DIMENSION_MULTIFRAME = 768
__C.TEST.HIDDEN_DIMENSION_SINGLEFRAME = 128

#####################################################################################################################################
#
# DEFINE THE MODELS' COMMON PARAMETERS 
#
#####################################################################################################################################

#
# Define learning rate, batch size and loss
#  

__C.TEST.LEARNING_RATE = 0.1
__C.TEST.BATCH_SIZE = 1
__C.TEST.LOSS = 'MSE'

#
# Define dataset descriptors
# Len_sequence : define the lenght of the sequence to predict
# Shuffle: indicates if the images have to be shuffle
# 
__C.TEST.LEN_SEQUENCES = 30
__C.TEST.SHUFFLE = 'False'

######################################################################################################################################






