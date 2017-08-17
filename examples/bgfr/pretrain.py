import os
import sys
import math
import shutil
import subprocess

import caffe
from caffe.model_libs import *
from google.protobuf import text_format

caffe_root = os.getcwd()

# Paths to lmdb files
train_data = 'data/bgfr_four_corpora/2016-02-10_cnnG_four-corpora-887/pretrain.train_lmdb'
test_data = 'data/bgfr_four_corpora/2016-02-10_cnnG_four-corpora-887/pretrain.val_lmdb'

# Organize current experiment
job_name = '2016-02-10_cnnG_four-corpora-887'
# Provide a name for model
model_name = 'Pretrain_{}'.format(job_name)
# Directory for storing .prototxt files
save_dir = os.path.join('models/BGFRNet', job_name, 'pretrain')
# Directory for storing snapshot models
snapshot_dir = os.path.join('models/BGFRNet', job_name, 'pretrain')
# Directory for storing bash and log files
job_dir = os.path.join('jobs/BGFRNet', job_name, 'pretrain')

# Model prototxt files
train_net_file = os.path.join(save_dir, 'train.prototxt')
test_net_file = os.path.join(save_dir, 'test.prototxt')

# Snapshot prefix
snapshot_prefix = os.path.join(snapshot_dir, model_name)

# Job bash file
job_file = os.path.join(job_dir, '{}.sh'.format(model_name))

# Some important hyperparameters
# Check https://github.com/BuddyGuard/RD_AI_FaceRecognition/wiki/Experiments 
# for more information
num_training_images = 65885
train_batch_size = 64
base_lr = 0.001
lr_low = base_lr / train_batch_size
lr_high = base_lr * train_batch_size
max_epochs = 100
num_iters_per_epoch = num_training_images / float(train_batch_size)
max_iter = num_iters_per_epoch * max_epochs
weight_decay = 0.0001
momentum = 0.9
num_test_images = 1836
test_batch_size = 64
test_iter = num_test_images/float(test_batch_size)
input_transform_scale = 0.00390625

# Solver parameters
solver_param = {
    # Train parameters
    'base_lr':base_lr,
    'lr_low': lr_low,
    'lr_high': lr_high,
    'lr_policy': 'bgfr',
    'weight_decay': weight_decay,
    'momentum': momentum,
    'max_iter': max_iter,
    'type': 'SGD',
    'display' : 100,
    'snapshot_after_train': True,
    'debug_info': False,
    # Test parameters
    'test_iter': test_iter,
    'test_interval': 300,
    'test_initialization': False,
    }

# Check if given data file exist
assert os.path.exists(train_data)
assert os.path.exists(test_data)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net
net = caffe.NetSpec()
net.data, net.label  = BGFRNetInput(data_file=train_data, 
                                    batch_size=train_batch_size, 
                                    scale=input_transform_scale,
                                    train=True,
                                    output_label=True)
BGFRNetBody(net, from_layer='data', train=True)

print net.to_proto()
