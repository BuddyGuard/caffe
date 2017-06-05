import os
import sys
import glob
import caffe
import numpy as np
import argparse
import operator
from collections import OrderedDict
from sklearn.cluster import KMeans

caffe.set_mode_gpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT - PRUNED - RETRAINED MODELS
#model_folders_filter = 'models/VGGNet/VOC0712/SSD_300x300_layer_indep_*_pruned'
#model_filter = 'VGG_VOC0712_SSD_300x300_layer_indep_*_pruned_iter_15000.caffemodel'
#clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Models')

# SSD VGGNet PASCAL LAYER WISE - PRUNED - RETRAINED MODELS 
#model_folders_filter = 'models/VGGNet/VOC0712/SSD_300x300_layer_wise_*_pruned'
#model_filter = 'VGG_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000.caffemodel'
#clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Models')

# SSD VGGNet - COCO - LAYER INDEPENDENT PRUNED - RETRAINED 15K - CLUSTERED 64
#model_folders_filter = 'models/VGGNet/coco/SSD_300x300_layer_indep_*_pruned'
#model_filter = 'VGG_coco_SSD_300x300_layer_indep_*_pruned_iter_15000.caffemodel'
#clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Independent_Pruned_Retrained_Clustered64_Models')
#num_centroids = 2**6 - 1

# SSD VGGNet - COCO - LAYER INDEPENDENT PRUNED - RETRAINED 15K - CLUSTERED 32
model_folders_filter = 'models/VGGNet/coco/SSD_300x300_layer_indep_*_pruned'
model_filter = 'VGG_coco_SSD_300x300_layer_indep_*_pruned_iter_15000.caffemodel'
clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Independent_Pruned_Retrained_Clustered32_Models')
num_centroids = 2**5 - 1

exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters

# Check Clustered Models path
if not os.path.isdir(clustered_models_path):
    os.makedirs(clustered_models_path)

# List all the pruned models
pruned_models_folder = glob.glob(model_folders_filter)
pruned_models_folder.sort()

# Start clustering each model
for model_folder in pruned_models_folder:
    pruned_model = glob.glob(os.path.join(model_folder, model_filter))[0]
    pruned_model_proto = os.path.join(model_folder, 'deploy.prototxt')
    clustered_model = os.path.splitext(os.path.basename(pruned_model))[0]
    clustered_model = os.path.join(clustered_models_path, '{}_clustered.caffemodel'.format(clustered_model))
    if os.path.isfile(clustered_model):
        continue
    pruned_net = caffe.Net(pruned_model_proto, pruned_model, caffe.TEST)
    clustered_net = caffe.Net(pruned_model_proto, pruned_model, caffe.TEST)
    print 'Clustering : {}'.format(pruned_model)
    for name, param in pruned_net.params.iteritems():
        weights_dict = dict()
        if name not in exclude_layers:
            # Collect all parameters in the layer into dictionary
            weights = param[0].data
            for n in xrange(weights.shape[0]):
                for c in xrange(weights.shape[1]):
                    for h in xrange(weights.shape[2]):
                        for w in xrange(weights.shape[3]):
                            if weights[n,c,h,w]:
                                key = '{}-{}-{}-{}-{}'.format(name, n, c, h, w)
                                if weights[n,c,h,w]:
                                    weights_dict[key] = weights[n,c,h,w]
            # Find minimum and maximum values
            min_val = weights.flatten().min()
            max_val = weights.flatten().max()
            # Initialize centroids
            centroids = np.linspace(min_val, max_val, num=num_centroids).reshape(-1, 1)
            # Training data
            nnz = len(weights_dict)
            X = np.array(weights_dict.values(), dtype=np.float32).reshape(-1, 1)
            print '\nClustering : {}'.format(name)
            # K-Means
            kmeans = KMeans(n_clusters=num_centroids, 
                            init=centroids, n_init=1, n_jobs=-1, verbose=1).fit(X)
            itr = 1
            while np.unique(kmeans.labels_).size != num_centroids:
                print 'Reinit itr = {}, labels={}'.format(itr, np.unique(kmeans.labels_).size)
                centroids = kmeans.cluster_centers_
                kmeans = KMeans(n_clusters=num_centroids, 
                                init=centroids, n_init=1, n_jobs=-1, verbose=1).fit(X)
                itr += 1
            # Update clustered net
            final_centroids = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_ 
            print 'Clustered {} : params={}, centroids={}, labels={}'.format(name, nnz, 
                                                                        final_centroids.size, np.unique(labels).size)
            for label, key in zip(labels, weights_dict.keys()):
                param_idx = key.strip('\n')
                param_idx = key.split('-')
                name = param_idx[0]
                n = int(param_idx[1])
                c = int(param_idx[2])
                h = int(param_idx[3])
                w = int(param_idx[4])
                clustered_net.params[name][0].data[n,c,h,w] = final_centroids[label]
    
    clustered_net.save(clustered_model)
    print 'Saved clustered model to ',clustered_model
print 'All done!'
