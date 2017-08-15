import os
import caffe
import sys

caffe.set_mode_cpu()

caffe_root = '/home/karthik/workspace/caffe'

usage = '''
    python total_params.py <proto> <caffemodel>
'''
if len(sys.argv) != 3:
    print usage

proto = sys.argv[1]
model = sys.argv[2]

net = caffe.Net(proto, model, caffe.TEST)

params_count = 0
for name, param in net.params.iteritems():
    for p in param:
		if len(p.data.shape) == 4:
			params_count += p.data.size 
params_count = float(params_count)/10**6

print '\nTotal parameters : {} Million'.format(params_count)
