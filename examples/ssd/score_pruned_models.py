import os
import numpy as np
import subprocess

caffe_root = '/home/karthik/workspace/caffe'

prune = False
cluster = True

# SSD VGGNet PASCAL LAYER INDEPENDENT
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'

# SSD VGGNet PASCAL LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'

# CLUSTERED SSD VGGNet PASCAL LAYER INDEPENDENT
#clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Clustering_Layer_Independent_Pruned_Models')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'
#prune_type = 'layer_indep'

# CLUSTERED SSD VGGNet PASCAL LAYER INDEPENDENT
#clustered_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Clustering_Layer_Wise_Pruned_Models')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'
#prune_type = 'layer_wise'

# SSD VGGNet COCO LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel')
#scoring_script = 'examples/ssd/score_ssd_coco_vggnet.py'
#gammas = np.arange(0.1, 2.1, 0.1)

# SSD ResNet PASCAL LAYER INDEPENDENT
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_resnet.py'
#prune_type = 'layer_indep'

# SSD ResNet PASCAL LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_ssd_pascal_resnet.py'

# CLUSTERED SSD ResNet PASCAL LAYER INDEPENDENT
#clustered_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Clustering_Layer_Independent_Pruned_Models')
#original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_resnet.py'
#prune_type = 'layer_indep'

# CLUSTERED SSD ResNet PASCAL LAYER WISE
clustered_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Clustering_Layer_Wise_Pruned_Models')
original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
scoring_script = 'examples/ssd/score_pruned_ssd_pascal_resnet.py'
prune_type = 'layer_wise'

if prune:
    models = os.listdir(pruned_models_path)
elif cluster:
    models = os.listdir(clustered_models_path)

models.sort()

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
        # Copy pruned model to original model location
        if prune:
            caffemodel = os.path.join(pruned_models_path, model)
        elif cluster:
            caffemodel = os.path.join(clustered_models_path, model)
        subprocess.call('cp {} {}'.format(caffemodel, original_model_name), shell=True)
        print 'Copied {} -> {}'.format(caffemodel, original_model_name)
        # Run score script
        if prune and prune_type == 'layer_indep':
            cmd = 'python {} {} {} {}'.format(scoring_script, 'prune', 'layer_indep', prune_percent)
        elif cluster and prune_type == 'layer_indep':
            cmd = 'python {} {} {} {}'.format(scoring_script, 'cluster', 'layer_indep', prune_percent)
        elif prune and prune_type == 'layer_wise':
            cmd = 'python {} {} {} {} {}'.format(scoring_script, 'prune', 'layer_wise', prune_percent, std_dev)
        elif cluster and prune_type == 'layer_wise':
            cmd = 'python {} {} {} {} {}'.format(scoring_script, 'cluster', 'layer_wise', prune_percent, std_dev)
        #print cmd
        subprocess.call(cmd, shell=True)

