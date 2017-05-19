import sys
import os
import glob
import numpy as np
from collections import OrderedDict
import caffe

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT - PRUNED - RETRAINED 10K - CLUSTERED MODELS
retrained_iter = '10000'
retrained_models_folders = glob.glob('models/VGGNet/VOC0712/SSD_300x300_layer_indep_*_pruned')
clustered_models_path = 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Models'
compressed_models_path = 'models/VGGNet/VOC0712/Layer_Independent_Pruned_Retrained_Clustered_Compressed_Models'

# SSD VGGNet PASCAL LAYER WISE - PRUNED - RETRAINED 15K - CLUSTERED MODELS
#retrained_iter = '15000'
#retrained_models_folders = glob.glob('models/VGGNet/VOC0712/SSD_300x300_layer_wise_*_pruned') 
#clustered_models_path = 'models/VGGNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Models'
#compressed_models_path = 'models/VGGNet/VOC0712/Layer_Wise_Pruned_Retrained_Clustered_Compressed_Models'

codebook_size = 2**8
ind_bits_size = 2**4

# Check Compressed Models path
if not os.path.isdir(compressed_models_path):
    os.makedirs(compressed_models_path)

# List all the clustered models
clustered_models = os.listdir(clustered_models_path)
clustered_models.sort()

def get_nnz(weights, bits):
    zcount = 0
    nnz = 0
    for w in weights:
        if w:
            nnz += 1
            zcount = 0
        else:
            zcount += 1
            if zcount % bits == 0:
                nnz += 1
                zcount = 0
    return nnz

def encode_to_relative_indexing(weights, bits):
    enc_val = []
    enc_ind = []
    zcount = 0
    for wt in weights:
        if wt:
            enc_val.append(wt)
            enc_ind.append(zcount)
            zcount = 0
        else:
            zcount += 1
            if zcount % bits == 0:          
                enc_val.append(0)
                enc_ind.append(bits-1)
                zcount = 0
    nnz = len(enc_val)
    enc_ind_len = (nnz-1)/2 + 1
    enc_ind_stream = np.zeros(enc_ind_len, dtype=np.uint8)
    enc_ind = np.array(enc_ind, dtype=np.uint8)
    enc_ind_stream += enc_ind[np.arange(0, nnz, 2)]
    if nnz % 2 != 0:
        enc_ind_stream[:-1] += enc_ind[np.arange(1, nnz, 2)] * bits
    else:
        enc_ind_stream += enc_ind[np.arange(1, nnz, 2)] * bits 
                                                        
    return enc_val, enc_ind_stream

def encode_spm_stream(spm_stream, nnz, bits):
    enc_spm_stream_size = (nnz - 1) / 2 + 1
    enc_spm_stream = np.zeros(enc_spm_stream_size, dtype=np.uint8)
    enc_spm_stream += spm_stream[np.arange(0, nnz, 2)]
    if nnz % 2 != 0:
        enc_spm_stream[:-1] += spm_stream[np.arange(1, nnz, 2)] * bits
    else:
        enc_spm_stream += spm_stream[np.arange(1, nnz, 2)] * bits
    return enc_spm_stream

for model in clustered_models:
    if not model.endswith('.caffemodel'):
        continue
    if not retrained_iter in model:
        print 'Skipping : {}. Retrained iteration does not match'.format(model)
        continue
    clustered_model = os.path.join(clustered_models_path, model)
    deploy_proto = ''
    for retrained_model_folder in retrained_models_folders:
        folder = model[model.find('SSD'):model.find('_iter')]
        if folder == os.path.basename(retrained_model_folder):
            deploy_proto = os.path.join(retrained_model_folder, 'deploy.prototxt')
    
    net = caffe.Net(deploy_proto, clustered_model, caffe.TEST)
    compressed_model = '{}.compressed_caffemodel'.format(os.path.splitext(model)[0])
    compressed_model = os.path.join(compressed_models_path, compressed_model)
    # Use binary file to save params
    fout = open(compressed_model, 'wb')
    # Get count of number of elements in relative index encoding
    print '\nGetting non-zero elements from each layer'
    nnzs = []
    for name, param in net.params.iteritems():
        for p in param:
            if len(p.data.shape) == 4:
                weights = p.data
                nnz = get_nnz(weights.flatten(), ind_bits_size)
                nnzs.append(nnz)
    # Write nnz to file
    nnzs = np.array(nnzs, dtype=np.uint32)
    nnzs.tofile(fout)
    # Encode params from each layer and write to file
    for name, param in net.params.iteritems():
        for p in param:
            if len(p.data.shape) == 4:
                weights = p.data
                codebook = OrderedDict()
                codebook[0] = 0
                pos = 1
                for val in np.unique(weights.flatten()):
                    if not val in codebook.keys():
                        codebook[val] = pos
                        pos += 1
                val_stream, ind_stream = encode_to_relative_indexing(weights.flatten(), ind_bits_size)
                spm_stream = np.zeros(len(val_stream), dtype=np.uint8)
                for pos, val in enumerate(val_stream):
                    spm_stream[pos] = codebook[val]
                #enc_spm_stream = encode_spm_stream(spm_stream, spm_stream.size, ind_bits_size)
                codebook_vals = np.array(codebook.keys(), dtype=np.float32)
                print 'Compressing {} : codebook={}, spm_stream={}, ind_stream={}'.format(name, codebook_vals.size,
                                                                                          spm_stream.size, ind_stream.size)
                codebook_vals.tofile(fout)
                spm_stream.tofile(fout)
                ind_stream.tofile(fout)
            else:
                print 'Compressing {} : bias={}'.format(name, p.data.size)
                bias = p.data.flatten().astype(np.float32)
                bias.tofile(fout)
    fout.close()
    print 'Saved compressed model at : {}'.format(compressed_model)
