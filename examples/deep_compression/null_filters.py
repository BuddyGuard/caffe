import os
import sys
import caffe
import numpy as np
from collections import OrderedDict

usage = """
    python null_filters.py <deplo<-proto> <caffemodel>
"""

if len(sys.argv) != 3:
    print usage
    sys.exit()

proto = sys.argv[1]
model = sys.argv[2]

net = caffe.Net(proto, model, caffe.TEST)

null_filters = OrderedDict()
for name, params in net.params.iteritems():
    null_filters_count = 0
    null_channels = 0
    for p in params:
        if len(p.data.shape) == 4:
            weights = p.data
            for i in range(weights.shape[0]):
                filter_data = weights[i,:,:,:]
                total_params_count = filter_data.size
                nonzero_params_count = np.count_nonzero(filter_data.flatten())
                zeros = total_params_count - nonzero_params_count
                if zeros == total_params_count:
                    null_filters_count += 1
            null_filters[name] = null_filters_count

for name, count in null_filters.iteritems():
    print '{} : {}'.format(name, count)
