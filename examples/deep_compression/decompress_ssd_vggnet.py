import sys
import os
import numpy as np
from collections import OrderedDict
import caffe

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT - PRUNED - RETRAINED 10K - CLUSTERED - COMPRESSED MODELS
retrained_models_folder_template = 'models/VGGNet/VOC0712/SSD_300x300_layer_indep_{}%_pruned'
compressed_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Compressed_Models')
decompressed_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Decompressed_Models')
prototxt = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300_layer_indep_40%_pruned/deploy.prototxt.txt')
exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters
codebook_size = 2**8
ind_bits_size = 2**4

def decode_relative_indexing(weights, spm_stream, ind_stream, codebook, nnz, ind_bits):
    # Init holders
    code = np.zeros(weights.size, np.uint8) 
    ind = np.zeros(nnz, np.uint8)
    # Get even positioned indices
    ind[np.arange(0, nnz, 2)] = ind_stream % ind_bits
    # Get odd positioned indices                                
    ind[np.arange(1, nnz, 2)] = ind_stream / ind_bits
    # Recover original weights array indices
    ind = np.cumsum(ind+1)-1
    # Assign back orignial indices
    code[ind] = spm_stream
    # Assign respective codes
    data = np.reshape(codebook[code], weights.shape)
    # Copy reovered data to weights
    np.copyto(weights, data)

# Check Decompressed Models path
if not os.path.isdir(decompressed_models_path):
    os.makedirs(decompressed_models_path)

# List all the clustered models
compressed_models = os.listdir(compressed_models_path)
compressed_models.sort()

def get_prune_percent(model):
    idx = model.find('%')
    return model[idx-2:idx]

for model in compressed_models:
    # Get pruned percentage of model
    prune_percent = get_prune_percent(model)
    # Deploy prototxt
    deploy_proto = os.path.join(retrained_models_folder_template.format(prune_percent), 'deploy.prototxt')
    # Set paths
    compressed_model = os.path.join(compressed_models_path, model)
    decompressed_model = '{}_decompressed.caffemodel'.format(os.path.splitext(model)[0])
    decompressed_model = os.path.join(decompressed_models_path, decompressed_model)
    # Init net
    net = caffe.Net(deploy_proto, caffe.TEST)
    # Get valid layers count
    layers = []
    for name, param in net.params.iteritems():
        if name not in exclude_layers:
            layers.append(name)
    # Open file
    fin = open(compressed_model, 'rb')
    # Read non-zero elements count 
    nnzs = np.fromfile(fin, dtype = np.uint32, count = len(layers))
    # Decode each layer params
    for pos, layer in enumerate(layers):
        print 'Reconstructing Layer : {}'.format(layer)
        # Read codebook 
        codebook = np.fromfile(fin, dtype = np.float32, count = codebook_size)
        # Read bias
        bias = np.fromfile(fin, dtype = np.float32, count = net.params[layer][1].data.size)
        # Copy bias to net
        np.copyto(net.params[layer][1].data, bias)
        # Read codebook indices
        spm_stream = np.fromfile(fin, dtype = np.uint8, count = nnzs[pos])
        # Read indices
        ind_stream = np.fromfile(fin, dtype = np.uint8, count = (nnzs[pos]-1) / 2+1)
        # Recover weights
        decode_relative_indexing(net.params[layer][0].data, spm_stream, ind_stream, codebook, nnzs[pos], ind_bits_size)
    fin.close()
    net.save(decompressed_model)
    print 'Saved decompressed model at : {}'.format(decompressed_model)
