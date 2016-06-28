#ifndef CAFFE_YOLO_DATA_LAYER_HPP_
#define CAFFE_YOLO_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
class YoloDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit YoloDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param){}

	virtual ~YoloDataLayer();

	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "YoloData"; }

	virtual inline int ExactNumBottomBlobs() const { return 0; }

	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	virtual void ShuffleImages();

	virtual void load_batch(Batch<Dtype>* batch);

	int Rand(int n);

	Dtype Constrain(Dtype min, Dtype max, Dtype value);

	void CropImage(const cv::Mat& cv_img, const int dx, const int dy,
			const int w, const int h, cv::Mat& cropped_img);

	void ReadBoxes(const string& label_path, vector<BoundingBox<Dtype> >& bboxes);

	void CorrectBoxes(vector<BoundingBox<Dtype> >& bboxes, const Dtype dx, const Dtype dy,
			const Dtype sx, const Dtype sy, const int flip);

	Dtype RandUniform(Dtype min, Dtype max);

	void FillTruthRegion(const string& filename, vector<Dtype>& truth, const int classes, const int side,
			const int flip, const float dx, const float dy, const float sx, const float sy);

	void LoadYoloData(const std::string& filename, const int height, const int width,
		    		const bool is_color, const float jitter, cv::Mat& cv_img, vector<Dtype>& truth);

	shared_ptr<Caffe::RNG> prefetch_rng_;

	int height_, width_, channels_, batch_size_;
	int mirror_;
	float saturation_;
	float exposure_;
	float jitter_;
	int num_of_predictions_;
	int side_, classes_, coords_;
	bool is_color_;
	string train_data_file_;

	int lines_id_;   //current image id
	vector<string> lines_;
};

}

#endif  // CAFFE_YOLO_DATA_LAYER_HPP_
