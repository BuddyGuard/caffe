import sys
import os
import glob
import numpy as np
from collections import OrderedDict
import caffe

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD ResNet PASCAL - LAYER-INDEPENDENT PRUNED - RETRAINED 15K - CLUSTERED - COMPRESSED MODELS
#retrained_models_folders = glob.glob('models/ResNet/VOC0712/SSD_300x300_layer_indep_*_pruned')
#compressed_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Compressed_Models')
#decompressed_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Decompressed_Models')

# SSD ResNet PASCAL - LAYER-WISE PRUNED - RETRAINED 15K - CLUSTERED - COMPRESSED MODELS
retrained_models_folders = glob.glob('models/ResNet/VOC0712/SSD_300x300_layer_wise_*_pruned')
compressed_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Compressed_Models')
decompressed_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Decompressed_Models')

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

for model in compressed_models:
    # Deploy prototxt
    deploy_proto = ''
    for retrained_model_folder in retrained_models_folders:
        folder = model[model.find('SSD'):model.find('_iter')]
        if folder == os.path.basename(retrained_model_folder):
            deploy_proto = os.path.join(retrained_model_folder, 'deploy.prototxt')
            break
    # Set paths
    compressed_model = os.path.join(compressed_models_path, model)
    decompressed_model = '{}_decompressed.caffemodel'.format(os.path.splitext(model)[0])
    decompressed_model = os.path.join(decompressed_models_path, decompressed_model)
    # Init net
    net = caffe.Net(deploy_proto, caffe.TEST)
    # Get list of conv layers
    layers = []
    for name, param in net.params.iteritems():
        for p in param:
            if len(p.data.shape) == 4:
                layers.append(name)
    # Open file
    fin = open(compressed_model, 'rb')
    # Read non-zero elements count of conv layers  
    nnzs = np.fromfile(fin, dtype = np.uint32, count = len(layers))
    # nnzs dict
    nnzs_dict = OrderedDict(zip(layers, nnzs))
    # Decode each layer params
    for name, param in net.params.iteritems():
        for pos, p in enumerate(param):
            if len(p.data.shape) == 4:
                print 'Reconstructing Layer : {}, shape={}'.format(name, p.data.shape)
                nnz = nnzs_dict[name]
                # Read codebook 
                codebook = np.fromfile(fin, dtype = np.float32, count = codebook_size)
                # Read codebook indices
                spm_stream = np.fromfile(fin, dtype = np.uint8, count = nnz)
                # Read indices
                ind_stream = np.fromfile(fin, dtype = np.uint8, count = (nnz-1) / 2+1)
                # Recover weights
                decode_relative_indexing(net.params[name][pos].data, spm_stream, 
                                         ind_stream, codebook, nnz, ind_bits_size)
            else:
                print 'Reconstructing Layer : {}, shape={}'.format(name, p.data.shape)
                # Read BN Mean, BN Variance, BN eps, Scale Multiplier, Scale Bias, Conv Bias 
                params = np.fromfile(fin, dtype = np.float32, count = net.params[name][pos].data.size)
                # Copy params to net
                np.copyto(net.params[name][pos].data, params) 
    fin.close()
    net.save(decompressed_model)
    print 'Saved decompressed model at : {}'.format(decompressed_model)
