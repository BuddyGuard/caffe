import os
import caffe
import numpy as np
import argparse
import operator
from collections import OrderedDict

caffe.set_mode_gpu()

caffe_root = '/home/karthik/workspace/caffe'
model = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt')
caffemodel = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel_bkp')
exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters
gamma = 0.7 

net = caffe.Net(model, caffemodel, caffe.TEST)

total_params = 0
prune_count = 0
removable_params = OrderedDict()
for name, param in net.params.iteritems():
     if name in exclude_layers:
        continue
     if len(param) == 2:
        weights = param[0].data
        if len(weights.shape) == 4:
            param_count = weights.shape[0]*weights.shape[1]*weights.shape[2]*weights.shape[3]
            total_params += param_count
            std = np.std(weights.flatten())
            count = np.where(np.absolute(weights.flatten()) < gamma*std)[0]
            prune_count += len(count)
            removable_params[name] = float(len(count))/param_count*100
            print '{} : Params < {} = {}'.format(name, std, len(count))
            count = np.where(np.absolute(weights) < gamma*std)
            for n, c, h, w in zip(count[0],count[1],count[2],count[3]):
                net.params[name][0].data[n,c,h,w] = 0.0  

for key in removable_params.keys():
    print '{} : {} % are prunable'.format(key, removable_params[key])

prune_percent = float(prune_count)/total_params*100
print '{} % of total parameters {} Million can be removed'.format(prune_percent, float(total_params)/1000000)

basename = os.path.basename(caffemodel)
pruned_net = '{}_cd_pruned_{}%.caffemodel'.format(os.path.splitext(basename)[0], int(prune_percent))

print 'Saving pruned model ', pruned_net
net.save(pruned_net)
