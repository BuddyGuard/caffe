#ifndef CAFFE_SIAMESE_DATA_LAYER_HPP_
#define CAFFE_SIAMESE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SiameseDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SiameseDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SiameseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SiameseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const {return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  shared_ptr<Caffe::RNG> prefetch_rng_;

  bool is_color_;
  int height_, width_, channels_, batch_size_;
  std::vector<std::string> filenames_;
  std::vector<int> datasetids_;
  std::vector<int> labels_;
  std::vector<int> labels_start_index_;
  std::vector<int> datasetids_start_index_;
 
}; 

} // namespace caffe

#endif // CAFFE_SIAMESE_DATA_LAYER_HPP_
