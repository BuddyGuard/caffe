from __future__ import print_function
import os
import sys
import math
import stat
import shutil
import subprocess

import caffe
from caffe.model_libs import *
from google.protobuf import text_format

caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

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
solver_file = os.path.join(save_dir, 'solver.prototxt')

# Snapshot prefix
snapshot_prefix = os.path.join(snapshot_dir, model_name)

# Job bash file
job_file = os.path.join(job_dir, '{}.sh'.format(model_name))

# Some important hyperparameters
# Check https://github.com/BuddyGuard/RD_AI_FaceRecognition/wiki/Experiments 
# for more information
# Train parameters
num_training_images = 65885
train_batch_size = 32
base_lr = 0.001
lr_low = base_lr / train_batch_size
lr_high = base_lr * train_batch_size
max_epochs = 100
num_iters_per_epoch = num_training_images / float(train_batch_size)
max_iter = int(num_iters_per_epoch * max_epochs)
snapshot_interval = 900
weight_decay = 0.0001
momentum = 0.9
# Test parameters
num_test_images = 1836
test_batch_size = 256
test_interval = 300
test_iter = int(num_test_images/float(test_batch_size))
input_transform_scale = 0.00390625

# Solver parameters
gpus = '0'
gpulist = gpus.split(',')
num_gpus = len(gpulist)
solver_mode = P.Solver.GPU
batch_size_per_device = train_batch_size
#accum_batch_size = train_batch_size # Change this if you experience GPU memory problems
#if num_gpus > 0:
#    batch_size_per_device = int(math.ceil(float(train_batch_size) / num_gpus))
#    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
#    solver_mode = P.Solver.GPU
#    device_id = int(gpulist[0])
solver_type = 'SGD'

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'lr_low': lr_low,
    'lr_high': lr_high,
    'lr_policy': 'bgfr',
    'weight_decay': weight_decay,
    'momentum': momentum,
    'max_iter': max_iter,
    'snapshot': snapshot_interval,
    'max_epochs': max_epochs,
    'type': solver_type,
    'solver_mode': solver_mode,
    'display' : 100,
    'snapshot_after_train': True,
    'debug_info': False,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': test_interval,
    'test_initialization': False,
    }
'''
solver_param = {
    # Train parameters
    'base_lr': 0.01,
    'lr_policy': "inv",
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'max_iter': 25000,
    'snapshot': 900,
    'gamma': 0.0001,
    'power': 0.75,
    'solver_mode': solver_mode,
    'display' : 100,
    'snapshot_after_train': True,
    'debug_info': False,
    # Test parameters
    'test_iter': [8],
    'test_interval': 300,
    'test_initialization': False,
    }
'''
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

with open(train_net_file, 'w') as f:
    print('name : "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net
net = caffe.NetSpec()
net.data, net.label = BGFRNetInput(data_file=test_data,
                                   batch_size=test_batch_size,
                                   scale=input_transform_scale,
                                   train=False,
                                   output_label=True)
BGFRNetBody(net, from_layer='data', train=False)

with open(test_net_file, 'w') as f:
    print('name : "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)


# Create solver
solver =  caffe_pb2.SolverParameter(train_net=train_net_file,
                                    test_net=[test_net_file],
                                    snapshot_prefix=snapshot_prefix,
                                    **solver_param)
with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

# Find most recent snapshot
max_iter = 0
for f in os.listdir(snapshot_dir):
    if f.endswith('.solverstate'):
        basename = os.path.splitext(f)[0]
        itr = int(basename.split('{}_iter_'.format(model_name))[1])
        if itr > max_iter:
            max_iter = itr

train_src_param = None
if resume_training:
    if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

# Create job file
with open(job_file, 'w') as f:
    f.write('cd {}\n'.format(caffe_root))
    f.write('./build/tools/caffe train \\\n')
    f.write('--solver="{}" \\\n'.format(solver_file))
    if train_src_param:
        f.write(train_src_param)
    if solver_param['solver_mode'] == P.Solver.GPU:
	f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
    else:
    	f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy this python script to job dir
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
    subprocess.call(job_file, shell=True)
