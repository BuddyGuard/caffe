#ifndef CAFFE_BOUNDING_BOX_HPP_
#define CAFFE_BOUNDING_BOX_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace caffe{

template <typename Dtype>
class BoundingBox {
public:
	BoundingBox(){}

	Dtype x;
	Dtype y;
	Dtype w;
	Dtype h;

};  // class BoundingBox

}  // namespace caffe

#endif  // CAFFE_BOUNDING_BOX_HPP_
