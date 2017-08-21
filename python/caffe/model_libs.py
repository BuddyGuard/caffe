import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def BGFRNetInput(data_file, batch_size, scale, train=True, 
                 output_label=True, backend=P.Data.LMDB):
    if train:
        kwargs = {
                  'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                  'transform_param': dict(scale=scale),
                  'data_param': dict(source=data_file, batch_size=batch_size,
                                     backend=backend),
                 }
    else:
        kwargs = {
                  'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                  'transform_param': dict(scale=scale),
                  'data_param': dict(source=data_file, batch_size=batch_size,
                                     backend=backend),
                 }
    if output_label and train:
        data, label = L.Data(name="train_data", ntop=2, **kwargs)
    elif output_label and not train:
        data, label = L.Data(name="test_data", ntop=2, **kwargs)    
    return [data, label]
       

def BGFRNetBody(net, from_layer, train=True):
    kwargs = dict()
    if train:
        kwargs = {'param':[dict(lr_mult=1, decay_mult=1), 
                           dict(lr_mult=2, decay_mult=0)],
                  'weight_filler': dict(type='xavier'),
                  'bias_filler': dict(type='constant', value=0)
                 }

    net.conv1 = L.Convolution(net[from_layer], kernel_size=7,
                              num_output=16, stride=1, pad=0, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2 = L.Convolution(net.pool1, kernel_size=6,
                              num_output=48, stride=1, pad=0, **kwargs)                            
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.pool2 = L.Pooling(net.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=3)

    net.conv3 = L.Convolution(net.pool2, kernel_size=5,
                              num_output=96, stride=1, pad=0, **kwargs)            
    net.relu3 = L.ReLU(net.conv3, in_place=True)

    net.fc1 = L.InnerProduct(net.relu3, num_output=128, **kwargs)
    net.relu4 = L.ReLU(net.fc1, in_place=True)

    from_layer = 'relu4'
    if train:
        net.drop1 = L.Dropout(net.relu4, dropout_ratio=0.5, in_place=True)
        from_layer = 'drop1'

    net.out = L.InnerProduct(net[from_layer], num_output=848, **kwargs)

    if train:
        net.loss = L.SoftmaxWithLoss(net.out, net.label)
    else:
        net.accuracy = L.Accuracy(net.out, net.label)
