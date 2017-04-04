import os
import sys
import caffe
import numpy as np
import argparse
import operator
from collections import OrderedDict

caffe.set_mode_gpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD RESNET PASCAL LAYER INDEPENDENT PRUNED
#prototxt = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/deploy.prototxt_bkp')
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruning')
#clustered_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Clustering_Layer_Independent_Pruned_Models')

# SSD RESNET PASCAL LAYER WISE PRUNED
prototxt = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/deploy.prototxt_bkp')
pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruning')
clustered_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Clustering_Layer_Wise_Pruned_Models')

# Check Clustered Models path
if not os.path.isdir(clustered_models_path):
    os.makedirs(clustered_models_path)

# Maximum number of iterations for clustering
max_iters = 1000

# List all the pruned models
pruned_models = os.listdir(pruned_models_path)
pruned_models.sort()

# Start clustering each model
for model in pruned_models:
    pruned_model = os.path.join(pruned_models_path, model)
    pruned_net = caffe.Net(prototxt, pruned_model, caffe.TEST)
    clustered_net = caffe.Net(prototxt, pruned_model, caffe.TEST)
    for name, param in pruned_net.params.iteritems():
        weights_dict = dict()
        for p in param:
            if len(p.data.shape) == 4:
                print 'Collecting Non-zeros params in ',name
                # Collect all parameters in the layer into dictionary
                weights = p.data
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
                centroids = np.linspace(min_val, max_val, num=256)
                # Initialize some temp variables
                iters = 0
                cost_sum = None
                final_centroids = None
                previous_distance = sys.float_info[0]
                current_distance = 0.0
                print 'Clustering Layer : ',name
                # Clustering weights
                while iters < max_iters:
                # Check convergence
                    if (np.fabs(previous_distance-current_distance)/previous_distance) < 0.01:
                        final_centroids = cents_wts_labels_dict
                        break
                    # Update previous distance
                    previous_distance = current_distance
                    current_distance = 0.0
                    centroids_dict = dict()
                    cents_wts_labels_dict = dict()
                    # Initialize containers for each centroid
                    for c in centroids:
                        centroids_dict[c] = []
                        cents_wts_labels_dict[c] = []
                    min_distances = []
                    # Assign each weight to nearest centroid
                    for key in weights_dict.keys():
                        wt = weights_dict[key]
                        distances = np.fabs(centroids - wt)
                        centroid = centroids[np.argmin(distances)]
                        min_distances.append(distances.min())
                        centroids_dict[centroid].append(wt)
                        cents_wts_labels_dict[centroid].append(key)
                    # Calculate new distance
                    current_distance = np.sum(np.asarray(min_distances))
                    # Update centroids
                    centroids = []
                    for centroid in centroids_dict.keys():
                        wts = centroids_dict[centroid]
                        if len(wts):
                            centroids.append(np.mean(np.asarray(wts)))
                        else:
                            centroids.append(centroid)
                    print 'Iter {} : J={}'.format(iters+1, current_distance)
                    iters += 1
                # Update  Weights in Clustered Net
                for centroid in final_centroids.keys():
                    keys = final_centroids[centroid]
                    for key in keys:
                        param_idx = key.strip('\n')
                        param_idx = key.split('-')
                        name = param_idx[0]
                        n = int(param_idx[1])
                        c = int(param_idx[2])
                        h = int(param_idx[3])
                        w = int(param_idx[4])
                        clustered_net.params[name][0].data[n,c,h,w] = centroid

    clustered_model = os.path.join(clustered_models_path,
                      '{}_clustered.caffemodel'.format(os.path.splitext(model)[0]))
    clustered_net.save(clustered_model)
    print 'Saved clustered model to ',clustered_model

print 'All done!'
  
