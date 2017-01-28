import argparse
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
caffe.set_mode_cpu()

"""
This function is taken from https://github.com/jay-mahadeokar/pynetbuilder/tree/master/netbuilder/tools
"""
def get_complexity(netspec=None, prototxt_file=None):
    # One of netspec, or prototxt_path params should not be None
    assert (netspec is not None) or (prototxt_file is not None)

    if netspec is not None:
        prototxt_file = _create_file_from_netspec(netspec)
    net = caffe.Net(prototxt_file, caffe.TEST)

    total_params = 0
    total_flops = 0

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file).read(), net_params)

    for layer in net_params.layer:
        if layer.name in net.params:

            params = net.params[layer.name][0].data.size
            # If convolution layer, multiply flops with receptive field
            # i.e. #params * datawidth * dataheight
            if layer.type == 'Convolution':  # 'conv' in layer:
                data_width = net.blobs[layer.name].data.shape[2]
                data_height = net.blobs[layer.name].data.shape[3]
                flops = net.params[layer.name][0].data.size * data_width * data_height
                # print >> sys.stderr, layer.name, params, flops
            else:
                flops = net.params[layer.name][0].data.size

            total_params += params
            total_flops += flops



    if netspec is not None:
        os.remove(prototxt_file)

    return total_params, total_flops
    
if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('net', type=str, help='Network prototxt')
    
    args = arg_parser.parse_args()
    
    params, flops = get_complexity(prototxt_file=args.net)
    
    print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    print 'Number of flops: ', (1.0 * flops *0.001) / 1000000.0, ' Billion'
