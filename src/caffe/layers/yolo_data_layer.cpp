#ifdef USE_OPENCV

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/bounding_box.hpp"

#include "caffe/layers/yolo_data_layer.hpp"

namespace caffe {

template <typename Dtype>
YoloDataLayer<Dtype>::~YoloDataLayer(){
	this->StopInternalThread();
}

template <typename Dtype>
void YoloDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top){

	YoloDataParameter yolo_data_param = this->layer_param_.yolo_data_param();

	// Necessary parameters
	CHECK(yolo_data_param.has_batch_size()) << "Batch size is not specified";
	CHECK(yolo_data_param.has_height()) << "Height is not specified";
	CHECK(yolo_data_param.has_width()) << "Width is not specified";
	CHECK(yolo_data_param.has_channels()) << "Channels is not specified";
	CHECK(yolo_data_param.has_train_data_file()) << "Train data file is not specified";
	CHECK(yolo_data_param.has_num_of_predictions()) << "Number of predictions is not specified";
	CHECK(yolo_data_param.has_side()) << "Side is not specified";
	CHECK(yolo_data_param.has_classes()) << "Classes is not specified";
	CHECK(yolo_data_param.has_coords()) << "Coords is not specified";
	CHECK(yolo_data_param.has_jitter()) << "Jitter is not specified";
	CHECK(this->transform_param_.has_scale()) << "Scale is not specified";


	height_ = yolo_data_param.height();
	width_ = yolo_data_param.width();
	CHECK(height_ == width_) << "Image height and width should be same";

	std::ifstream infile(yolo_data_param.train_data_file().c_str());
	CHECK(infile.good()) << "Failed to open training data file "
	      << yolo_data_param.train_data_file() << std::endl;

	batch_size_ = yolo_data_param.batch_size();
	channels_ = yolo_data_param.channels();
	train_data_file_ = yolo_data_param.train_data_file();
	num_of_predictions_ = yolo_data_param.num_of_predictions();
	side_ = yolo_data_param.side();
	classes_ = yolo_data_param.classes();
	coords_ = yolo_data_param.coords();
	jitter_ = yolo_data_param.jitter();

	// Optional parameters
	mirror_ = yolo_data_param.has_mirror() ? yolo_data_param.mirror() : 0;
	saturation_ = yolo_data_param.has_saturation() ? yolo_data_param.saturation() : 0;
	exposure_ = yolo_data_param.has_exposure() ? yolo_data_param.exposure() : 0;

	// Load image paths
	CHECK(infile.is_open()) << "Unable to open the training data info file";
	for(std::string line; std::getline(infile, line);){
		lines_.push_back(line);
	}

	// Shuffle training data
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	ShuffleImages();

	// Read an image to initialize top blob
	lines_id_ = 0;
	is_color_ = channels_ ? true : false;
	cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_],
			height_, width_, is_color_);
	CHECK(cv_img.data) << "No image data" << lines_[lines_id_];

	// Infer data blob shape using data transformer
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
	this->transformed_data_.Reshape(top_shape);

	batch_size_ = batch_size_ > 0 ? batch_size_ : 1;
	top_shape[0] = batch_size_;

	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	    this->prefetch_[i].data_.Reshape(top_shape);
	}
	top[0]->Reshape(top_shape);

	// Label
	vector<int> label_shape(4);
	label_shape[0] = batch_size_;
	label_shape[1] = 1;
	label_shape[2] = 1;
	label_shape[3] = side_ * side_ * (classes_ + coords_ + 1);

	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	    this->prefetch_[i].label_.Reshape(label_shape);
	}
}

template <typename Dtype>
void YoloDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void YoloDataLayer<Dtype>::CropImage(const cv::Mat& cv_img, const int dx, const int dy,
		const int w, const int h, cv::Mat& cropped_img){

	cropped_img = cv::Mat::zeros(h, w, cv_img.type());

	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			for(int k = 0; k < cv_img.channels(); k++){
				int r = i + dy;
				int c = j + dx;
				if(r >= 0 && r < cv_img.rows && c >= 0 && c < cv_img.cols) {
					cropped_img.at<cv::Vec3b>(i, j)[k] = cv_img.at<cv::Vec3b>(r, c)[k];
				}
			}
		}
	}
}

template <typename Dtype>
int YoloDataLayer<Dtype>::Rand(int n) {
  CHECK(prefetch_rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void  YoloDataLayer<Dtype>::ReadBoxes(const string& label_path, vector<BoundingBox<Dtype> >& bboxes){

	std::ifstream infile(label_path.c_str());
	CHECK(infile.good()) << "Failed to open label file "
			<< label_path << std::endl;

	for(std::string line; std::getline(infile, line);){
		BoundingBox<Dtype> bbox;
		std::istringstream bbox_data(line);
		bbox_data >> bbox.id;
		bbox_data >> bbox.x;
		bbox_data >> bbox.y;
		bbox_data >> bbox.w;
		bbox_data >> bbox.h;

		bbox.left = bbox.x - bbox.w/2;
		bbox.right = bbox.x + bbox.w/2;
		bbox.top = bbox.y - bbox.h/2;
		bbox.bottom = bbox.y + bbox.h/2;

		bboxes.push_back(bbox);
	}

	infile.close();

}

template <typename Dtype>
Dtype YoloDataLayer<Dtype>::Constrain(Dtype min, Dtype max, Dtype value){

	if(value < min) return min;
	if(value > max) return max;
	return value;
}

template <typename Dtype>
void YoloDataLayer<Dtype>::CorrectBoxes(vector<BoundingBox<Dtype> >& bboxes, const Dtype dx, const Dtype dy,
		const Dtype sx, const Dtype sy, const int flip){

	for(int i = 0; i < bboxes.size(); ++i){
		bboxes[i].left   = bboxes[i].left  * sx - dx;
		bboxes[i].right  = bboxes[i].right * sx - dx;
		bboxes[i].top    = bboxes[i].top   * sy - dy;
		bboxes[i].bottom = bboxes[i].bottom * sy - dy;

		if(flip){
			Dtype swap = bboxes[i].left;
		    bboxes[i].left = 1.0 - bboxes[i].right;
		    bboxes[i].right = 1.0 - swap;
		}

		bboxes[i].left = Constrain(0.0, 1.0, bboxes[i].left);
		bboxes[i].right = Constrain(0.0, 1.0, bboxes[i].right);
		bboxes[i].top = Constrain(0.0, 1.0, bboxes[i].top);
        bboxes[i].bottom = Constrain(0.0, 1.0, bboxes[i].bottom);

        bboxes[i].x = (bboxes[i].left + bboxes[i].right)/2;
        bboxes[i].y = (bboxes[i].top + bboxes[i].bottom)/2;
        bboxes[i].w = bboxes[i].right - bboxes[i].left;
        bboxes[i].h = bboxes[i].bottom - bboxes[i].top;

        bboxes[i].w = Constrain(0.0, 1.0, bboxes[i].w);
		bboxes[i].h = Constrain(0.0, 1.0, bboxes[i].h);
	}

}


template <typename Dtype>
void YoloDataLayer<Dtype>::FillTruthRegion(const string& filename, vector<Dtype>& truth, const int classes, const int side,
		const int flip, const float dx, const float dy, const float sx, const float sy){

	// TODO: Generalize getting label path names
	string label_file = filename;
	string find_word = "JPEGImages";
	label_file.replace(label_file.find(find_word), find_word.length(), "labels");
	find_word = ".jpg";
	label_file.replace(label_file.find(find_word), find_word.length(), ".txt");

	// Read bounding boxes data
	vector<BoundingBox<Dtype> > bboxes;
	ReadBoxes(label_file, bboxes);

	// Shuffle boxes order
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(bboxes.begin(), bboxes.end(), prefetch_rng);

	// Correct boxes
	CorrectBoxes(bboxes, dx, dy, sx, sy, flip);

	// Fill truth blob
	Dtype x,y,w,h;
	int id;

	for (int i = 0; i < bboxes.size(); ++i) {
		x =  bboxes[i].x;
	    y =  bboxes[i].y;
	    w =  bboxes[i].w;
	    h =  bboxes[i].h;
	    id = bboxes[i].id;

	    if (w < .01 || h < .01) continue;

	    int col = (int)(x*side_);
	    int row = (int)(y*side_);

	    x = x*side_ - col;
	    y = y*side_ - row;

	    int index = ( col + row * side_)*(classes_ + coords_ + 1);
	    if (truth[index]) continue;
	    truth[index++] = 1;

	    if (id < classes) truth[index+id] = 1;
	    index += classes;

	    truth[index++] = x;
	    truth[index++] = y;
	    truth[index++] = w;
	    truth[index++] = h;
	}
}

template <typename Dtype>
void YoloDataLayer<Dtype>::LoadYoloData(const std::string& filename, const int height, const int width,
		const bool is_color, const float jitter, cv::Mat& cv_img, vector<Dtype>& truth){

	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_org = cv::imread(filename, cv_read_flag);

	CHECK(cv_img_org.data) << "No image data " << filename;

	int org_height = cv_img_org.rows;
	int org_width = cv_img_org.cols;

	int dw = org_width * jitter;
	int dh = org_height * jitter;

	float pleft, pright, ptop, pbottom;
	caffe_rng_uniform(1, float(-dw), float(dw), &pleft);
	caffe_rng_uniform(1, float(-dw), float(dw), &pright);
	caffe_rng_uniform(1, float(-dh), float(dh), &ptop);
	caffe_rng_uniform(1, float(-dh), float(dh), &pbottom);

	int swidth = org_width - pleft - pright;
	int sheight = org_height - ptop - pbottom;

	int flip = Rand(2);
	cv::Mat cropped_img;

	CropImage(cv_img_org, pleft, ptop, swidth, sheight, cropped_img);

	cv::Mat resized_img;
	cv::resize(cropped_img, resized_img, cv::Size(width, height));

	if(flip) cv::flip(resized_img, cv_img, 1);
	else cv_img = resized_img;

	float sx = float(swidth)/org_width;
	float sy = float(sheight)/org_height;
	float dx = float(pleft)/swidth;
	float dy = float(ptop)/sheight;

	FillTruthRegion(filename, truth, classes_, side_, flip, dx, dy, 1./sx, 1./sy);

}


template <typename Dtype>
void YoloDataLayer<Dtype>::load_batch(Batch<Dtype>* batch){

	CPUTimer batch_timer;
	batch_timer.Start();
	Dtype read_time = 0;
	Dtype trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	// Reshape images according to the first image of the batch
	cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_],
			height_, width_, is_color_);
	CHECK(cv_img.data) << "No image data " << lines_[lines_id_];

	// Data transformer
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
	this->transformed_data_.Reshape(top_shape);

	// Reshape batch
	top_shape[0] = batch_size_;
	batch->data_.Reshape(top_shape);
	vector<int> label_shape(4);
	label_shape[0] = batch_size_;
	label_shape[1] = 1;
	label_shape[2] = 1;
	label_shape[3] = side_ * side_ * (classes_ + coords_ + 1);
	batch->label_.Reshape(label_shape);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();

	caffe_set(batch->data_.count(), Dtype(0), prefetch_data);
	caffe_set(batch->label_.count(), Dtype(0), prefetch_label);


	// Read image, apply random crop, resize; Read bounding boxes from file and
	// transform it w.r.t to transformed image
	const int lines_size = lines_.size();
	int truth_len = side_ * side_ * (classes_ + coords_ + 1);
	for(int item_id = 0; item_id < batch_size_; ++item_id){
		timer.Start();
		CHECK_GT(lines_size, lines_id_);

		cv::Mat cv_img;
		vector<Dtype> truth(truth_len);
		LoadYoloData(lines_[lines_id_], height_, width_, is_color_,
				jitter_, cv_img, truth);

		read_time += timer.MicroSeconds();
		timer.Start();

		//Set image data to blob
		int data_offset = batch->data_.offset(item_id);
		this->transformed_data_.set_cpu_data(prefetch_data + data_offset);
		this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

		// Set label data to blob
		for(int j = 0; j < truth.size(); ++j){
			int offset = batch->label_.offset(item_id, 0, 0, j);
			prefetch_label[offset] = truth[j];
		}

		trans_time += timer.MicroSeconds();

		// Next iteration
		lines_id_++;
		if(lines_id_ >= lines_size){
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			ShuffleImages();
		}
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(YoloDataLayer);
REGISTER_LAYER_CLASS(YoloData);

}// namespace caffe
#endif  // USE_OPENCV
