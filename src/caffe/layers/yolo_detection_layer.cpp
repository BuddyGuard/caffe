#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/yolo_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void YoloDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

	YoloDetectionParameter yolo_param = this->layer_param_.yolo_detection_param();

	CHECK(yolo_param.has_classes()) << "classes is not specified";

	CHECK(yolo_param.has_coords()) << "coords is not specified";

	CHECK(yolo_param.has_rescore()) << "rescore is not specified";

	CHECK(yolo_param.has_side()) << "side is not specified";

	CHECK(yolo_param.has_num()) << "num is not specified";

	CHECK(yolo_param.has_sqrt()) << "sqrt is not specified";

	CHECK(yolo_param.has_no_object_scale()) << "no_object_scale is not specified";

	CHECK(yolo_param.has_object_scale()) << "object_scale is not specified";

	CHECK(yolo_param.has_class_scale()) << "class_scale is not specified";

	CHECK(yolo_param.has_coord_scale()) << "coord_scale is not specified";

	classes_ = yolo_param.classes();
	coords_ = yolo_param.coords();
	rescore_ = yolo_param.rescore();
	side_ = yolo_param.side();
	num_ = yolo_param.num();
	sqrt_ = yolo_param.sqrt();
	no_object_scale_ = yolo_param.no_object_scale();
	object_scale_ = yolo_param.object_scale();
	class_scale_ = yolo_param.class_scale();
	coord_scale_ = yolo_param.coord_scale();
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

	int label_count = bottom[0]->num()* side_ * side_ * (1 + coords_ + classes_);

	CHECK_EQ(bottom[1]->count(), label_count)
		<< "Label Dimension is not correct";

	vector<int> loss_shape(0);
	top[0]->Reshape(loss_shape);

	diff_.ReshapeLike(*bottom[0]);

	caffe_set(diff_.count(), Dtype(0), diff_.mutable_cpu_data());
}

template <typename Dtype>
BoundingBox<Dtype> YoloDetectionLayer<Dtype>::GetBoundingBox(const Dtype* blob, int start_idx){
	BoundingBox<Dtype> bbox;
	bbox.x = blob[start_idx+0];
	bbox.y= blob[start_idx+1];
	bbox.w = blob[start_idx+2];
	bbox.h = blob[start_idx+3];
	return bbox;
}

template <typename Dtype>
Dtype YoloDetectionLayer<Dtype>::BoxOverlap(const Dtype x1, const Dtype w1, const Dtype x2, const Dtype w2){
	Dtype l1 = x1 - w1/2;
	Dtype l2 = x2 - w2/2;
	Dtype left = l1 > l2 ? l1 : l2;
	Dtype r1 = x1 + w1/2;
	Dtype r2 = x2 + w2/2;
	Dtype right = r1 < r2 ? r1 : r2;

	return right - left;
}

template <typename Dtype>
Dtype YoloDetectionLayer<Dtype>::BoxIntersection(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	Dtype w = BoxOverlap(a.x, a.w, b.x, b.w);
	Dtype h = BoxOverlap(a.y, a.h, b.y, b.h);

	if(w < 0 || h < 0){
		return Dtype(0);
	}

	Dtype area = w * h;
	return area;
}

template <typename Dtype>
Dtype YoloDetectionLayer<Dtype>::BoxUnion(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	Dtype i = BoxIntersection(a, b);
	Dtype u = a.w * a.h + b.w * b.h - i;
	return u;
}

template <typename Dtype>
Dtype YoloDetectionLayer<Dtype>::BoxRMSE(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2) + pow(a.w - b.w, 2) + pow(a.h-b.h, 2));
}

template <typename Dtype>
Dtype YoloDetectionLayer<Dtype>::BoxIOU(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	return BoxIntersection(a, b) / BoxUnion(a, b);
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

	int batch_size = bottom[0]->shape(0);
	int locations = side_ * side_;
	Dtype avg_iou = 0;
	Dtype avg_cat = 0;
	Dtype avg_allcat = 0;
	Dtype avg_obj = 0;
	Dtype avg_anyobj = 0;
	Dtype cost = 0;
	int count = 0;

	const Dtype* output = bottom[0]->cpu_data();
	const Dtype* truth = bottom[1]->cpu_data();

	Dtype* diff = diff_.mutable_cpu_data();

	for(int b = 0; b < batch_size; ++b){
		for(int i = 0; i < locations; ++i){
			int truth_index = bottom[1]->offset(b, 0, 0, i * (1 + coords_ + classes_));
			int is_obj = truth[truth_index];
			for(int j = 0; j < num_; ++j){
				int p_index = bottom[0]->offset(b, locations * classes_ + i * num_ + j);
				diff[p_index] = no_object_scale_ * (-output[p_index]);
				cost += no_object_scale_ * pow(output[p_index], 2);
				avg_anyobj += output[p_index];
			}

			int best_index = -1;
			Dtype best_iou = 0;
			Dtype best_rmse = 20;

			if(!is_obj){
				continue;
			}

			int class_index = bottom[0]->offset(b, i * classes_);

			for(int j = 0; j < classes_; ++j){
				diff[class_index+j] = class_scale_ * (truth[truth_index+1+j] - output[class_index+j]);
				cost += class_scale_ * pow(truth[truth_index+1+j] - output[class_index+j], 2);
				if(truth[truth_index + 1 + j]){
					avg_cat += output[class_index+j];
				}
				avg_allcat += output[class_index+j];
			}

			BoundingBox<Dtype> bbox_truth = GetBoundingBox(truth, truth_index+1+classes_);
			bbox_truth.x /= side_;
			bbox_truth.y /= side_;

			for(int j = 0; j < num_; ++j){
				int box_index = bottom[0]->offset(b, locations*(classes_ + num_) + (i * num_ + j) * coords_);
				BoundingBox<Dtype> bbox_out = GetBoundingBox(output, box_index);
				bbox_out.x /= side_;
				bbox_out.y /= side_;

				if(sqrt_){
					bbox_out.w =  bbox_out.w * bbox_out.w;
					bbox_out.h =  bbox_out.h * bbox_out.h;
				}

				Dtype iou = BoxIOU(bbox_out, bbox_truth);
				Dtype rmse = BoxRMSE(bbox_out, bbox_truth);

				if(best_iou > 0 || iou > 0){
					if(iou > best_iou){
						best_iou = iou;
						best_index = j;
					}
				}else{
					if(rmse < best_rmse){
						best_rmse = rmse;
						best_index = j;
					}
				}
			}

			int box_index = bottom[0]->offset(b, locations*(classes_ + num_) + (i * num_ + best_index) * coords_);
			int tbox_index = truth_index + 1 + classes_;

			BoundingBox<Dtype> bbox_out = GetBoundingBox(output, box_index);
			bbox_out.x /= side_;
			bbox_out.y /= side_;

			if(sqrt_){
				bbox_out.w = bbox_out.w * bbox_out.w;
				bbox_out.h = bbox_out.h * bbox_out.h;
			}

			Dtype iou = BoxIOU(bbox_out, bbox_truth);

			int p_index = bottom[0]->offset(b, locations * classes_ + i * num_ + best_index);
			cost -= no_object_scale_ * pow(output[p_index], 2);
			cost += object_scale_ * pow(1-output[p_index], 2);
			avg_obj += output[p_index];
			diff[p_index] = object_scale_ * (1 - output[p_index]);

			if(rescore_){
				diff[p_index] = object_scale_ * (iou - output[p_index]);
			}

			diff[box_index+0] = coord_scale_*(truth[tbox_index + 0] - output[box_index + 0]);
			diff[box_index+1] = coord_scale_*(truth[tbox_index + 1] - output[box_index + 1]);
			diff[box_index+2] = coord_scale_*(truth[tbox_index + 2] - output[box_index + 2]);
			diff[box_index+3] = coord_scale_*(truth[tbox_index + 3] - output[box_index + 3]);

			if(sqrt_){
				diff[box_index+2] = coord_scale_*(sqrt(truth[tbox_index + 2]) - output[box_index + 2]);
				diff[box_index+3] = coord_scale_*(sqrt(truth[tbox_index + 3]) - output[box_index + 3]);
		    }

			cost += pow(1-iou, 2);
			avg_iou += iou;
			++count;
		}
	}

	top[0]->mutable_cpu_data()[0] = Dtype(cost) / batch_size;


	LOG(INFO) << "Detection Avg IOU: " << Dtype(avg_iou)/count << ", Pos Cat: " << Dtype(avg_cat)/count << ", All Cat: " <<
			avg_allcat/(count*classes_) << ", Pos Obj: " << Dtype(avg_obj)/count << ", Any Obj: " <<
			avg_anyobj/(batch_size*locations*num_) << ", Object count: " << count;
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

	caffe_cpu_axpby(bottom[0]->count(), Dtype(1), diff_.cpu_data(), Dtype(0), bottom[0]->mutable_cpu_diff());

	/*
	Dtype min = 0;
	Dtype max = 0;
	for(int k=0; k<bottom[0]->count(); k++)
	{
		if(bottom[0]->cpu_diff()[k] < min) min = bottom[0]->cpu_diff()[k];
		if(bottom[0]->cpu_diff()[k] > max) max = bottom[0]->cpu_diff()[k];
	}
	LOG(INFO) << "detection.delta.min = " << min;
	LOG(INFO) << "detection.delta.max = " << max;
	*/
}

#ifdef CPU_ONLY
STUB_GPU(YoloDetectionLayer);
#endif

INSTANTIATE_CLASS(YoloDetectionLayer);
REGISTER_LAYER_CLASS(YoloDetection);


}
