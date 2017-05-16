import sys
import os
import numpy as np
from collections import OrderedDict
import caffe

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT - PRUNED - RETRAINED 10K - CLUSTERED MODELS
retrained_models_folder_template = 'models/VGGNet/VOC0712/SSD_300x300_layer_indep_{}%_pruned'
clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Models')
decompressed_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Decompressed_Models')
exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

def get_prune_percent(model):
    idx = model.find('%')
    return model[idx-2:idx]

# List clustered models
clustered_models = os.listdir(clustered_models_path)
clustered_models.sort()

# List decompressed models
decompressed_models = os.listdir(decompressed_models_path)
decompressed_models.sort()

models_diff = OrderedDict()

for clustered_model, decompressed_model in zip(clustered_models, decompressed_models):
    # Get pruned percentage of model
    prune_percent = get_prune_percent(clustered_model)
    # Deploy prototxt
    deploy_proto = os.path.join(retrained_models_folder_template.format(prune_percent), 'deploy.prototxt')
    if clustered_model != decompressed_model.replace('_decompressed', ''):
        print 'Model names does not match'
        print 'Clustered model : {}'.format(clustered_model)
        print 'Decompressed model : {}'.format(decompressed_model)
        continue
    clustered_model = os.path.join(clustered_models_path, clustered_model)
    decompressed_model = os.path.join(decompressed_models_path, decompressed_model)
    clustered_net = caffe.Net(deploy_proto, clustered_model, caffe.TEST)
    decompressed_net = caffe.Net(deploy_proto, decompressed_model, caffe.TEST)
    total_weights_diff = 0
    total_bias_diff = 0
    for name, param in clustered_net.params.iteritems():
        if name in exclude_layers:
            continue
        clustered_weights = param[0].data
        clustered_bias = param[1].data
        decompressed_weights = decompressed_net.params[name][0].data
        decompressed_bias =  decompressed_net.params[name][1].data
        total_weights_diff += np.sum(clustered_weights) - np.sum(decompressed_weights) 
        total_bias_diff += np.sum(clustered_bias) - np.sum(decompressed_bias)
    models_diff[clustered_model] = {'decompressed_model':decompressed_model,
                                    'weights_diff':total_weights_diff,
                                    'bias_diff':total_bias_diff}

for key in models_diff.keys():
    print '\nClustered model    : {}'.format(key)
    print 'Decompressed model : {}'.format(models_diff[key]['decompressed_model'])
    print 'Weights diff       : {}'.format(models_diff[key]['weights_diff'])
    print 'Bias diff          : {}'.format(models_diff[key]['bias_diff'])
