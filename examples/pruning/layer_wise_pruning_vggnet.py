import os
import caffe
import shutil
import numpy as np
import argparse
import operator
from collections import OrderedDict

caffe.set_mode_gpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet VOC0712
#model = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt_bkp')
#caffemodel = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel_bkp')
#save_model_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruning')

# SSD VGGNet MS COCO
#model = os.path.join(caffe_root, 'models/VGGNet/coco/SSD_300x300/deploy.prototxt_bkp')
#caffemodel = os.path.join(caffe_root, 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel_bkp')
#save_model_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Wise_Pruning')

# VOC0712_CDP
#model = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/SSD_300x300/deploy.prototxt_bkp')
#caffemodel = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/SSD_300x300/VGG_VOC0712CDP_SSD_300x300_iter_60000.caffemodel_bkp')
#save_model_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/Layer_Independent_Pruning')

# SSD - VGG - VOC0712CDP - LAYER WISE PRUNING
model = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/SSD_300x300/deploy.prototxt_bkp')
caffemodel = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/SSD_300x300/VGG_VOC0712CDP_SSD_300x300_iter_60000.caffemodel_bkp')
save_model_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/Layer_Wise_Pruning')
gammas = np.arange(0.2, 2.2, 0.2)

exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

if os.path.isdir(save_model_path):
    shutil.rmtree(save_model_path)

os.makedirs(save_model_path)

for gamma in gammas:
    net = caffe.Net(model, caffemodel, caffe.TEST)

    total_params = 0
    prune_count = 0
    removable_params = OrderedDict()
    print 'Pruning parameters with abs value < {}% of Std dev in each layer'.format(gamma*100)
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
    pruned_net = '{}_{}_stddev_{}%_pruned.caffemodel'.format(os.path.splitext(basename)[0], gamma, int(prune_percent))
    
    save_model_as = os.path.join(save_model_path, pruned_net)
    print 'Saving pruned model ', save_model_as
    net.save(save_model_as)
