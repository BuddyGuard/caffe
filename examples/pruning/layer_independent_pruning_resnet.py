import os
import caffe
import numpy as np
import argparse
import operator
from collections import OrderedDict

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'
model = os.path.join(caffe_root, 'models/ResNet/coco/SSD_300x300/deploy.prototxt_bkp')
caffemodel = os.path.join(caffe_root, 'models/ResNet/coco/SSD_300x300/ResNet_coco_SSD_300x300_iter_400000.caffemodel_bkp')
save_models_path = os.path.join(caffe_root, 'models/ResNet/Layer_Independent_Pruning')
exclude_layers = ['conv4_3_norm'] # Skip this layer's parameters
pruning_coeffs = np.arange(0.1, 1, 0.1) # Percentage of parameters to be removed 

def save_indexed_params():
    net = caffe.Net(model, caffemodel, caffe.TEST)
    params_count = 0
    params_file = open('params.txt', 'w')
    params_idx_file = open('params_idx.txt', 'w')
    for name, param in net.params.iteritems(): 
        #print '{} : No. of params = {}'.format(name, len(param))
        for p in param:
            if len(p.data.shape) == 4:
                print 'Processing {} : {}'.format(name, p.data.shape)
                weights = p.data                
                for n in xrange(weights.shape[0]):
                    for c in xrange(weights.shape[1]):
                        for h in xrange(weights.shape[2]):
                            for w in xrange(weights.shape[3]):
                                key = '{}-{}-{}-{}-{}'.format(name, n, c, h, w)
                                params_file.write('{}\n'.format(weights[n,c,h,w]))
                                params_idx_file.write('{}\n'.format(key))
                                params_count += 1
            else:
                #print 'Not Processing {} : {}'.format(name, p.data.shape)
                prod = 1
                for s in p.shape:
                    prod *= s
                params_count += prod
                
    params_file.close()
    params_idx_file.close()
    return params_count
        
def sort_params(params_count, percent):
    params_file = open('params.txt', 'r')
    params = []
    print 'Loading params from file!'
    for param in params_file:
        param = param.strip('\n')
        params.append(abs(float(param)))

    print 'Arg sorting!'
    sorted_idxs = np.argsort(params)
    
    del params
        
    prune_count = int(params_count * percent)
    
    prune_idxs = sorted_idxs[:prune_count]
    
    del sorted_idxs
    
    print 'Loading params idx from file!'
    params_idx_file = open('params_idx.txt', 'r')
    params_idx = []
    for idx in params_idx_file:
        idx = idx.strip('\n')
        params_idx.append(idx)
        
    print 'Saving sorted params idx to file!'
    sorted_idx_file = open('sorted_params_idx.txt', 'w')
    for idx in prune_idxs:
        sorted_idx_file.write('{}\n'.format(params_idx[idx]))
    
    del params_idx
        
    sorted_idx_file.close()            
        
def prune_network(params_count, percent):
    net = caffe.Net(model, caffemodel, caffe.TEST)
    
    print 'Number of parameters : {} Million'.format(float(params_count)/1000000)
    prune_count = int(params_count * percent)                        
    print 'Number of parameters to be pruned : {} Million'.format(float(prune_count)/1000000)
    
    prune_params_idx_file = open('sorted_params_idx.txt', 'r')
    
    for param_idx in prune_params_idx_file:
        param_idx = param_idx.strip('\n')
        param_idx = param_idx.split('-')
        name = param_idx[0]
        n = int(param_idx[1])
        c = int(param_idx[2])
        h = int(param_idx[3])
        w = int(param_idx[4])
        net.params[name][0].data[n,c,h,w] = 0.0
        
    if os.path.isfile('sorted_params_idx.txt'):
        os.remove('sorted_params_idx.txt')
                    
    basename = os.path.basename(caffemodel)
    pruned_net = '{}_{}%_pruned.caffemodel'.format(os.path.splitext(basename)[0], int(percent*100))
    pruned_net = os.path.join(save_models_path, pruned_net)
    net.save(pruned_net)
    print 'Saved pruned model to ',pruned_net
    
def abs_diff(pruned_model):
    net = caffe.Net(model, caffemodel, caffe.TEST)
    net_pruned = caffe.Net(model, pruned_model, caffe.TEST)
    abs_diffs = OrderedDict()
    # Absolute sum difference between original and pruned layer
    for name, param in net.params.iteritems():
        if name in exclude_layers:
            continue
        if len(param) == 2:
            if len(param[0].shape) == 4:
                org_sum = np.sum(param[0].data)
                prune_sum = np.sum(net_pruned.params[name][0].data)
                abs_diffs[name] = org_sum - prune_sum
                
    for key in abs_diffs.keys():
        print '{} : {}'.format(key, abs_diffs[key])
        
def count_zeros(pruned_model):
    net_pruned = caffe.Net(model, pruned_model, caffe.TEST)
    zeros_count = OrderedDict()
    zeros_percentage = OrderedDict()
    for name, param in net_pruned.params.iteritems():
        for p in param:
            if len(p.data.shape) == 4:
                print 'Counting zeros in {} : {}'.format(name, p.data.shape)
                num_of_params = p.data.size;
                non_zero_params = np.count_nonzero(p.data)
                zeros = num_of_params - non_zero_params
                zeros_count[name] = zeros
                zeros_percentage[name] = float(zeros)/num_of_params*100
    
    print 'Number of zeros in each layer'
    for key in zeros_count.keys():
        print '{} : No. of zeros = {}, Percentage = {}%'.format(key, zeros_count[key], zeros_percentage[key])
                
        
    
if __name__=='__main__':

        
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('net', nargs='?', help='Pruned caffemodel', )
    args_parser.add_argument('--prune', help='Perform pruning on given network', action='store_true')
    args_parser.add_argument('--diff', help='Compare absolute difference between original and pruned network', action='store_true')
    args_parser.add_argument('--zeros', help='Count number of zeros in each layer', action='store_true')
    
    args = args_parser.parse_args()
    if args.prune:
        params_count = save_indexed_params()
        for percent in pruning_coeffs:
            sort_params(params_count, percent)
            prune_network(params_count, percent)
         
        if os.path.isfile('params.txt'):
            os.remove('params.txt')
	if os.path.isfile('params_idx.txt'):
            os.remove('params_idx.txt')

    elif args.diff:
        abs_diff(args.net)
    elif args.zeros:
        count_zeros(args.net)
                        
