#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif // USE_OPENCV
#include <vector>
#include <fstream>
#include <string>

#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/siamese_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void SiameseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    SiameseDataParameter siamese_data_param = this->layer_param_.siamese_data_param();

    // Check necessary parameters
    CHECK(siamese_data_param.has_batch_size()) << "Batch size is not specified";
    CHECK(siamese_data_param.has_height()) << "Input image height is not specified";
    CHECK(siamese_data_param.has_width()) << "Input image width is not specified";
    CHECK(siamese_data_param.has_channels()) << "Input image channels is not specified";
    CHECK(siamese_data_param.has_data_file()) << "Training data file is not provided";
    CHECK(siamese_data_param.has_corpora_root()) << "Corpora root path is not provided";
    
    height_ = siamese_data_param.height();
    width_ = siamese_data_param.width();
    channels_ = siamese_data_param.channels();
    batch_size_ = siamese_data_param.batch_size();
    std::string corpora_root = siamese_data_param.corpora_root();

    std::ifstream infile(siamese_data_param.data_file().c_str());
    CHECK(infile.good()) << "Failed to open training data file "
        << siamese_data_param.data_file();
    
    // Read file contents
    std::string filename;
    int datasetid;
    int label;
    while (infile >> filename >> datasetid >> label ) {
        filenames_.push_back(corpora_root+"/"+filename);
        datasetids_.push_back(datasetid);
        labels_.push_back(label);
    }
    LOG(INFO) << "Total images : " << filenames_.size();
    
    // Collect start indices
    int prev_label = 0;
    int prev_datasetid = 0;
    labels_start_index_.push_back(0);
    datasetids_start_index_.push_back(0);
    for (int i=0; i < filenames_.size(); ++i ) {
        if (prev_label != labels_[i]) {
            prev_label = labels_[i];
            labels_start_index_.push_back(i);
        }
        if (prev_datasetid != datasetids_[i]) {
            prev_datasetid = datasetids_[i];
            datasetids_start_index_.push_back(i);
        }
    }
    labels_start_index_.push_back(filenames_.size());
    datasetids_start_index_.push_back(filenames_.size());

    // Read an image to initialize top blob
    is_color_ = false; 
    cv::Mat cv_img = ReadImageToCVMat(filenames_[0], height_,
                            width_, is_color_);
    CHECK(cv_img.data) << "Could not load " << filenames_[0];
    // Infer data blob shape using data transformer
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    batch_size_ = batch_size_ > 0 ? batch_size_ : 1;
    top_shape[0] = batch_size_;
    top_shape[1] = 2;

    for (int i=0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(top_shape);
    }

    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
    
    // Label shape
    vector<int> label_shape(1, batch_size_);
    top[1]->Reshape(label_shape);
    for (int i=0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
    }
}

template <typename Dtype>
SiameseDataLayer<Dtype>::~SiameseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SiameseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
   
    // Data transformer
    cv::Mat cv_img = ReadImageToCVMat(filenames_[0],
                            height_, width_, is_color_);
    CHECK(cv_img.data) << "Could not load " << filenames_[0];
    
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    top_shape[0] = batch_size_;
    top_shape[1] = 2;
    // Reshape batch according to the batch_size.
    batch->data_.Reshape(top_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();
 
    for (int item_id = 0; item_id < batch_size_; ++item_id) {
        timer.Start();
        // Pick a random index
        int rand_idx = caffe::caffe_rng_rand() % filenames_.size();
        // Pick positive or negative pair randomly
        int pos_pair = caffe::caffe_rng_rand() % 2;
        //LOG(INFO) << "pos_pair : " << pos_pair;
        int pair_idx; 
        int current_label = labels_[rand_idx];
        if (pos_pair) {
            int search_start_idx = labels_start_index_[current_label];
            int search_end_idx = labels_start_index_[current_label+1] - 1;
            //LOG(INFO) << "search_start_idx : " << search_start_idx;
            //LOG(INFO) << "search_end_idx : " << search_end_idx;
            do { 
                pair_idx = search_start_idx + caffe::caffe_rng_rand() % (search_end_idx - search_start_idx);
            }
            while (rand_idx == pair_idx);
        } else {
            int datasetid = datasetids_[rand_idx];
            int search_start_idx = datasetids_start_index_[datasetid];
            int search_end_idx = datasetids_start_index_[datasetid+1] - 1;
            //LOG(INFO) << "search_start_idx : " << search_start_idx;
            //LOG(INFO) << "search_end_idx : " << search_end_idx;
            do {
                pair_idx = search_start_idx + caffe::caffe_rng_rand() % (search_end_idx - search_start_idx);
            }
            while (labels_[rand_idx] == labels_[pair_idx]);
        }
        //LOG(INFO) << "rand_idx : " << rand_idx << ", filename : " << filenames_[rand_idx];
        //LOG(INFO) << "pair_idx : " << pair_idx << ", filename : " << filenames_[pair_idx];
        // Read pair of images
        cv::Mat cv_img1 = ReadImageToCVMat(filenames_[rand_idx], 
                            height_, width_, is_color_);
        CHECK(cv_img1.data) << "Could not load " << filenames_[rand_idx];
        cv::Mat cv_img2 = ReadImageToCVMat(filenames_[pair_idx], 
                            height_, width_, is_color_);
        CHECK(cv_img2.data) << "Could not load " << filenames_[pair_idx];
        read_time += timer.MicroSeconds();

        // Set first image
        timer.Start();
        int offset = batch->data_.offset(item_id, 0);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_img1, &(this->transformed_data_));
        // Set second image
        offset = batch->data_.offset(item_id, 1);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_img2, &(this->transformed_data_));
        // Set label
        prefetch_label[item_id] = pos_pair;

        trans_time += timer.MicroSeconds();        
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SiameseDataLayer);
REGISTER_LAYER_CLASS(SiameseData);

} // namespace caffe
