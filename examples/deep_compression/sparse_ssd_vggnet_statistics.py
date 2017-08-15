import os
import sys
import glob
import caffe
import numpy as np
from collections import OrderedDict

caffe.set_mode_cpu()

# SSD VGG - VOC0712 - LAYER INDEP PRUNED - RETRAINED 15K
#model_folders_filter = 'models/VGGNet/VOC0712/SSD_300x300_layer_wise_*_pruned'
#model_filter = 'VGG_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000.caffemodel'
#model_folders = glob.glob(model_folders_filter)

# SSD VGG - VOC0712 - LAYER INDEP PRUNED - RETRAINED 15K - CLUSTERED 256
#odel_folders_root = 'models/VGGNet/VOC0712'
#lustered_models_path = 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Models'
#odel_filter = 'VGG_VOC0712_SSD_300x300_layer_indep_*_pruned_iter_15000_clustered.caffemodel'
#nc_ind_bit_size = 2**4
#lustered = True
#ill_compress = True

# SSD VGG - VOC0712 - LAYER WISE PRUNED - RETRAINED 15K - CLUSTERED 256
#model_folders_root = 'models/VGGNet/VOC0712'
#clustered_models_path = 'models/VGGNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Models'
#model_filter = 'VGG_VOC0712_SSD_300x300_layer_wise_*_pruned_iter_15000_clustered.caffemodel'
#enc_ind_bit_size = 2**4
#clustered = True
#will_compress = True

# SSD VGG - VOC0712CDP - LAYER INDEP PRUNED - RETRAINED 15K
#model_folders_filter = 'models/VGGNet/VOC0712CDP/SSD_300x300_layer_indep_*_pruned'
#model_filter = 'VGG_VOC0712CDP_SSD_300x300_layer_indep_*_pruned_iter_15000.caffemodel'
#model_folders = glob.glob(model_folders_filter)
#clustered = False
#will_compress = False

# SSD VGG - VOC0712CDP - LAYER INDEP PRUNED - RETRAINED 15K - CLUSTERED 256
model_folders_root = 'models/VGGNet/VOC0712CDP'
clustered_models_path = 'models/VGGNet/VOC0712CDP/Layer_Independent_Pruned_Retrained_Clustered256_Models'
model_filter = 'VGG_VOC0712CDP_SSD_300x300_layer_indep_*_pruned_iter_15000_clustered.caffemodel'
enc_ind_bit_size = 2**4
clustered = True
will_compress = True

def get_sparse_layer_elements_sizes(weights, bits):
    nnz = 0
    zcount = 0
    for wt in weights:
        if wt:
            zcount = 0
            nnz += 1
        else:
            zcount += 1
            if zcount % bits == 0:          
                zcount = 0
                nnz +=1
  
    enc_ind_len = (nnz-1)/2 + 1
    # Bytes of each data type
    per_float_bytes = np.arange(1).astype(np.float32).nbytes 
    per_uint_bytes = np.arange(1).astype(np.uint8).nbytes
    
    # Size of elements of sparse layer
    weights_size = (per_float_bytes * np.unique(weights).size)/float(1024**2)
    enc_indices_size = (per_uint_bytes * enc_ind_len)/float(1024**2)
    spm_code_size = (per_uint_bytes * nnz)/float(1024**2)
    
    return weights_size, enc_indices_size, spm_code_size

def get_stats(model, proto):
    print '\nModel : {}'.format(model)
    net = caffe.Net(proto, model, caffe.TEST)
    total = OrderedDict()
    nonzeros = OrderedDict()
    zeros_count = OrderedDict()
    zeros_percentage = OrderedDict()
    unique = OrderedDict()
    all_wts_size = 0
    all_bias_size = 0
    all_enc_ind_size = 0
    all_spm_code_size = 0
    for name, param in net.params.iteritems():
        for p in param:
            if len(p.data.shape) == 4:
                weights = p.data
                num_of_params = weights.size
                non_zero_params = np.count_nonzero(weights.flatten())
                zeros = num_of_params - non_zero_params
                total[name] = num_of_params
                nonzeros[name] = non_zero_params
                zeros_count[name] = zeros
                zeros_percentage[name] = float(zeros)/num_of_params*100
                unique[name] = np.unique(weights.flatten()).size
                if will_compress:
                    wts_size, enc_ind_size, spm_code_size = get_sparse_layer_elements_sizes(weights.flatten(), enc_ind_bit_size)
                    all_wts_size += wts_size
                    all_enc_ind_size += enc_ind_size
                    all_spm_code_size += spm_code_size
            else:
                bias = p.data
                if will_compress:
                    per_float_bytes = np.arange(1).astype(np.float32).nbytes
                    all_bias_size += (per_float_bytes * bias.flatten().size)/float(1024**2)
    return {'layers':total.keys(), 'total':total.values(), 'zeros':zeros_count.values(),
            'nonzeros':nonzeros.values(), 'unique':unique.values(),
            'weights_size':all_wts_size, 'bias_size':all_bias_size,
            'enc_ind_size':all_enc_ind_size, 'spm_code_size':all_spm_code_size}

# Dict for storing stats of each model
stats = OrderedDict()

if clustered:
    models = glob.glob(os.path.join(clustered_models_path, model_filter))
    models.sort()
    for model in models:
        model_folder = model[model.find('SSD'):model.find('_iter')]
        proto = os.path.join(model_folders_root, model_folder, 'deploy.prototxt')
        stats[model] = get_stats(model, proto)
else:
    model_folders = glob.glob(model_folders_filter)
    model_folders.sort()
    for model_folder in model_folders:
        if not glob.glob(os.path.join(model_folder, model_filter)):
            continue
        proto = os.path.join(model_folder, 'deploy.prototxt')
        model = glob.glob(os.path.join(model_folder, model_filter))[0]
        stats[model] = get_stats(model, proto)

for key in stats.keys():
    print "\ndata['model'] = '{}'".format(os.path.basename(key))
    print "data['layers'] = {}".format(stats[key]['layers'])
    print "data['total'] = {}".format(stats[key]['total'])
    print "data['zeros'] = {}".format(stats[key]['zeros'])
    print "data['nonzeros'] = {}".format(stats[key]['nonzeros'])
    print "data['unique'] = {}".format(stats[key]['unique'])
    if will_compress:
        print "data['weigths_size'] = {}".format(stats[key]['weights_size'])
        print "data['bias_size'] = {}".format(stats[key]['bias_size'])
        print "data['enc_ind_size'] = {}".format(stats[key]['enc_ind_size'])
        print "data['spm_code_size'] = {}".format(stats[key]['spm_code_size'])
