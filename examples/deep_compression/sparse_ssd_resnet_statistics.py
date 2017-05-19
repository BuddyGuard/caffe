import os
import sys
import glob
import caffe
import numpy as np
from collections import OrderedDict

caffe.set_mode_cpu()

# SSD RESNET VOC0712 LAYER INDEP PRUNED - RETRAINED 15K - CLUSTERED 256
#model_folders_filter = 'models/ResNet/VOC0712/SSD_300x300_layer_indep_*_pruned'
#model_filter = 'ResNet_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000.caffemodel'
#model_folders = glob.glob(model_folders_filter)
#clustered_models_path = 'models/ResNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Models'
#clustered = True

# SSD RESNET VOC0712 - LAYER WISE PRUNED
model_folders_filter = 'models/ResNet/VOC0712/Layer_Wise_Pruning'
model_filter = 'ResNet_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000.caffemodel'
pruned_models_path = 'models/ResNet/VOC0712/Layer_Wise_Pruning'
pruned = True
clustered = False


# SSD RESNET VOC0712 - LAYER WISE PRUNED - RETRAINED 15K - CLUSTERED 256
#model_folders_filter = 'models/ResNet/VOC0712/SSD_300x300_layer_wise_*_pruned'
#model_filter = 'ResNet_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000.caffemodel'
#model_folders = glob.glob(model_folders_filter)
#clustered_models_path = 'models/ResNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Models'
#clustered = True


exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

if clustered:
    clustered_models = os.listdir(clustered_models_path)
    clustered_models.sort()

model_folders = glob.glob(model_folders_filter)
model_folders.sort()

stats = OrderedDict()

for model_folder in model_folders:
    if clustered:
        for clustered_model in clustered_models:
            extract_folder = clustered_model[clustered_model.find('SSD'):clustered_model.find('_iter')]
            if os.path.basename(model_folder) == extract_folder:
                model = os.path.join(clustered_models_path, clustered_model)
    else: 
        model = glob.glob(os.path.join(model_folder, model_filter))[0]
    proto = os.path.join(model_folder, 'deploy.prototxt')
    print '\nModel : {}'.format(model)
    net = caffe.Net(proto, model, caffe.TEST)
    total = OrderedDict()
    nonzeros = OrderedDict()
    zeros_count = OrderedDict()
    zeros_percentage = OrderedDict()
    unique = OrderedDict()
    for name, param in net.params.iteritems():
        if name in exclude_layers:
            continue
        for p in param:
            if len(p.data.shape) == 4:
                weights = p.data
                if len(weights.shape) == 4:
                    num_of_params = weights.size
                    non_zero_params = np.count_nonzero(weights.flatten())
                    zeros = num_of_params - non_zero_params
                    total[name] = num_of_params
                    nonzeros[name] = non_zero_params
                    zeros_count[name] = zeros
                    zeros_percentage[name] = float(zeros)/num_of_params*100
                    unique[name] = np.unique(weights.flatten()).size
    stats[model] = {'layers':total.keys(), 'total':total.values(), 'zeros':zeros_count.values(),
                    'nonzeros':nonzeros.values(), 'unique':unique.values()}
    
for key in stats.keys():
    print "\ndata['model'] = '{}'".format(os.path.basename(key))
    print "data['layers'] = {}".format(stats[key]['layers'])
    print "data['total'] = {}".format(stats[key]['total'])
    print "data['zeros'] = {}".format(stats[key]['zeros'])
    print "data['nonzeros'] = {}".format(stats[key]['nonzeros'])
    print "data['unique'] = {}".format(stats[key]['unique'])
