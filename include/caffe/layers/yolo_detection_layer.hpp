#ifndef CAFFE_YOLO_DETECTION_LAYER_HPP_
#define CAFFE_YOLO_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/bounding_box.hpp"

namespace caffe{

template<typename Dtype>
class YoloDetectionLayer : public Layer<Dtype> {
public:
	explicit YoloDetectionLayer(const LayerParameter& param)
		: Layer<Dtype>(param){}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {return "YoloDetection";}

	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	//		const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

	//virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
	//		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

	BoundingBox<Dtype> GetBoundingBox(const Dtype* blob, int start_idx);

	Dtype BoxIOU(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b);

	Dtype BoxIntersection(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b);

	Dtype BoxUnion(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b);

	Dtype BoxOverlap(const Dtype x1, const Dtype w1, const Dtype x2, const Dtype w2);

	Dtype BoxRMSE(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b);

	Blob<Dtype> diff_;

	int classes_;
	int coords_;
	bool rescore_;
	int side_;
	int num_;
	bool sqrt_;
	float no_object_scale_;
	float object_scale_;
	float class_scale_;
	float coord_scale_;

	Dtype avg_loss_;
};

} // namespace caffe

#endif // CAFFE_YOLO_DETECTION_LAYER_HPP_
