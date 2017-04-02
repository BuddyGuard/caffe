import os
import numpy as np
import subprocess

caffe_root = '/home/karthik/workspace/caffe'

# SSD VGGNet PASCAL LAYER INDEPENDENT
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Independent_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'

# SSD VGGNet PASCAL LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/VOC0712/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_pruned_ssd_pascal_vggnet.py'

# SSD VGGNet COCO LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/VGGNet/coco/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel')
#scoring_script = 'examples/ssd/score_ssd_coco_vggnet.py'
#gammas = np.arange(0.1, 2.1, 0.1)

# SSD ResNet PASCAL LAYER INDEPENDENT
pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Independent_Pruning')
original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
scoring_script = 'examples/ssd/score_pruned_ssd_pascal_resnet.py'
prune_type = 'layer_indep'

# SSD ResNet PASCAL LAYER WISE
#pruned_models_path = os.path.join(caffe_root, 'models/ResNet/VOC0712/Layer_Wise_Pruning')
#original_model_name = os.path.join(caffe_root, 'models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_60000.caffemodel')
#scoring_script = 'examples/ssd/score_ssd_pascal_resnet.py'

pruned_models = os.listdir(pruned_models_path)
pruned_models.sort()

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
		# Copy pruned model to original model location
		pruned_model = os.path.join(pruned_models_path, model)
		subprocess.call('cp {} {}'.format(pruned_model, original_model_name), shell=True)
		print 'Copied {} -> {}'.format(pruned_model, original_model_name)
		# Run score script
		if prune_type == 'layer_indep':
			cmd = 'python {} {} {}'.format(scoring_script, 'layer_indep', prune_percent)
		else:
			cmd = 'python {} {} {} {}'.format(scoring_script, 'layer_wise', prune_percent, std_dev)
		print cmd
		subprocess.call(cmd, shell=True)

