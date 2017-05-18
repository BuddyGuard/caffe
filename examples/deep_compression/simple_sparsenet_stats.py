import sys
import numpy as np
import caffe

usage='''
Usage:
        python simple_sparsenet_stats.py <deploy-prototxt> <caffemodel>
'''

if len(sys.argv) != 3:
    print usage
    sys.exit()

deploy_prototxt = sys.argv[1]
caffemodel = sys.argv[2]

net = caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)

for name, params in net.params.iteritems():
    for p in params:
        if len(p.data.shape) == 4:
            weights = p.data.flatten()
            nnz = np.count_nonzero(weights)
            zeros = weights.size - nnz
            print '{} : nonzeros = {}, zeros={}'.format(name, nnz, zeros)
