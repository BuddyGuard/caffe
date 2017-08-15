import os
import sys
import caffe
import numpy as np
import argparse
import operator
from collections import OrderedDict

caffe.set_mode_gpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT
prototxt = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt_bkp')
clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Clustering_Layer_Independent_Pruned_Models')
exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

# SSD VGGNet PASCAL LAYER WISE
#rototxt = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt_bkp')
#runed_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruning')
#lustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Clustering_Layer_Wise_Pruned_Models')
#xclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

# List all clustered models
clustered_models = os.listdir(clustered_models_path)
clustered_models.sort()

for model in clustered_models:
    clustered_model = os.path.join(clustered_models_path, model)
    print 'Clustered model : {}'.format(clustered_model)
    clustered_net = caffe.Net(prototxt, clustered_model, caffe.TEST)
    for name, param in clustered_net.params.iteritems():
        if name not in exclude_layers:
            weights = param[0].data.flatten()
            unique_items = np.unique(weights).size
            print '{} : {}'.format(name, unique_items)

