import os
import re
import numpy as np
import subprocess

caffe_root = '/home/karthik/workspace/caffe'


# SSD VGGNet - COCO - LAYER INDEPENDENT PRUNED - RETRAINED 15K - CLUSTERED 16
clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Independent_Pruned_Retrained_Clustered16_Models')
original_model_name = os.path.join(caffe_root, 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel')
scoring_script = 'examples/ssd/score_pruned_ssd_coco_vggnet.py'
itr = 15000
prune_type = 'layer_indep'
prune = False
cluster = True
centroids = 16
decompressed = False

if prune:
    models = os.listdir(pruned_models_path)
elif cluster and not decompressed:
    models = os.listdir(clustered_models_path)
elif cluster and decompressed:
    models = os.listdir(decompressed_models_path)

if itr:
    temp_models = []
    for model in models:
        if str(itr) in model:
            temp_models.append(model)
    models = temp_models

models.sort()

print models

for model in models:
    if '.caffemodel' in model:
        if prune_type == 'layer_indep':
            key_str = '_pruned'
            base_idx = model.find(key_str)
            prune_percent = model[base_idx-3:base_idx-1]
        elif prune_type == 'layer_wise':
            key_str = '_stddev_'
            base_idx = model.find(key_str)
            std_dev = model[base_idx-3:base_idx]
            prune_percent = model[base_idx+len(key_str):base_idx+len(key_str)+2] 
            prune_percent = re.sub('\%$', '', prune_percent)
        # Copy pruned model to original model location
        if prune:
            caffemodel = os.path.join(pruned_models_path, model)
        elif cluster and not decompressed:
            caffemodel = os.path.join(clustered_models_path, model)
        elif cluster and decompressed:
            caffemodel = os.path.join(decompressed_models_path, model)
        subprocess.call('cp {} {}'.format(caffemodel, original_model_name), shell=True)
        print 'Copied {} -> {}'.format(caffemodel, original_model_name)
        # Run score script
        if prune and prune_type == 'layer_indep':
            cmd = 'python {} --pruned --pruned_type={} --prune_percent={}'.format(scoring_script, 'layer_indep', prune_percent)
        elif cluster and not decompressed and prune_type == 'layer_indep':
            cmd = 'python {} --clustered --centroids={} --prune_type={} --prune_percent={}'.format(scoring_script,  centroids, 'layer_indep', prune_percent)
        elif cluster and decompressed and prune_type == 'layer_indep':
            cmd = 'python {} --clustered --centroids={} --prune_type={} --prune_percent={} --decompressed'.format(scoring_script, centroids, 'layer_indep', prune_percent)
        
        elif prune and prune_type == 'layer_wise':
            cmd = 'python {} --pruned --pruned_type={} --prune_percent{} --std_dev={}'.format(scoring_script, prune_type, prune_percent, std_dev)
        elif cluster and not decompressed and prune_type=='layer_wise':
            cmd = 'python {} --clustered --centroids={} --prune_type={} --prune_percent={} --std_dev={}'.format(scoring_script, centroids, prune_type, prune_percent, std_dev)
        elif cluster and decompressed and prune_type=='layer_wise':
            cmd = 'python {}  --clustered --decompressed --centroids={} --prune_type={} --prune_percent={} --std_dev={}'.format(scoring_script, centroids, prune_type, prune_percent, std_dev)
        print cmd
        subprocess.call(cmd, shell=True)

