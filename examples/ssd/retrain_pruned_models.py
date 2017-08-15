import os
import numpy as np
import subprocess

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT
pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruning')
retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_vggnet.py'
prune_type = 'layer_indep'
selective_retraining = True
max_itr = 60000
step_itr = 40000
base_map = 0.72
maps = [0.723342, 0.721431, 0.720934,0.722612,0.726926,0.729182,0.728059,0.7260629,0.00170724]

# SSD VGGNet PASCAL LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruning')
#retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_vggnet.py'
#prune_type = 'layer_wise'

# SSD ResNet PASCAL LAYER INDEPENDENT
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruning')
#retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_resnet.py'
#prune_type = 'layer_indep'

# SSD RESNET - VOC0712 -  LAYER WISE PRUNED MODELS - RETRAINING
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruning')
#retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_resnet.py'
#prune_type = 'layer_wise'

# SSD - VGGNet - VOC0712CDP - LAYER INDEP PRUNED MODELS - RETRAIN 15K
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/Layer_Independent_Pruning')
#retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_cdp_vggnet.py'
#prune_type = 'layer_indep'

# SSD - VGGNet - VOC0712CDP - LAYER WISE PRUNED MODELS - RETRAIN 15K
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712CDP/Layer_Wise_Pruning')
#retraining_script = 'examples/ssd/retrain_pruned_ssd_pascal_cdp_vggnet.py'
#prune_type = 'layer_wise'

pruned_models = os.listdir(pruned_models_path)
pruned_models.sort()

if selective_retraining:
    remove_models = []
    for pos, i in enumerate(maps):
        if i > base_map:
            remove_models.append(pruned_models[pos])
    for model in remove_models:
        pruned_models.remove(model)

for model in pruned_models:
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
        
        pruned_model = os.path.join(pruned_models_path, model)
        # Run score script
        if prune_type == 'layer_indep':
            cmd = 'python {} --prune_type={} --prune_percent={} --pruned_model={} --max_itr={} --step_itr={}'.format(retraining_script, 'layer_indep', prune_percent, pruned_model, max_itr, step_itr)
        else:
            cmd = 'python {} {} {} {} {}'.format(retraining_script, 'layer_wise', prune_percent, std_dev, pruned_model)
        subprocess.call(cmd, shell=True)

