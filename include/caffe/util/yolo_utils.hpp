#ifndef CAFFE_YOLO_UTILS_HPP_
#define CAFFE_YOLO_UTILS_HPP_

#include <vector>

#include "caffe/blob.hpp"

namespace caffe{

template <typename Dtype>
class BoundingBox {
public:
	BoundingBox(){}
	int id;
	Dtype x;
	Dtype y;
	Dtype w;
	Dtype h;
	Dtype left;
	Dtype right;
	Dtype top;
	Dtype bottom;

};  // class BoundingBox

template<class ForwardIt>
size_t max_index(ForwardIt first, ForwardIt last) {
    if(first == last) return -1;
    ForwardIt largest = first;
    size_t largest_index = 0;
    ++first;
    for(size_t i = 1; first != last; ++first, ++i) {
        if(*largest < *first) {
            largest = first;
            largest_index = i;
        }
    }
    return largest_index;
}

class YoloBox {
   public:
    // Pay attention: x, y contain not the top left coords, but
    // coordinates of center
    float x;
    float y;
    float width;
    float height;

    float intersectArea(const YoloBox &other) const {
        float w = overlap(this->x, this->width, other.x, other.width);
        float h = overlap(this->y, this->height, other.y, other.height);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    float unionArea(const YoloBox &other) const {
        float i = this->intersectArea(other);
        return this->width * this->height + other.width * other.height - i;
    }

    float iou(const YoloBox &other) const {
        return this->intersectArea(other) / this->unionArea(other);
    }

   private:
    float overlap(float x1, float w1, float x2, float w2) const {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }
};

template <typename Dtype>
BoundingBox<Dtype> get_bounding_box(const Dtype* blob, int start_idx){
	BoundingBox<Dtype> bbox;
	bbox.x = blob[start_idx+0];
	bbox.y= blob[start_idx+1];
	bbox.w = blob[start_idx+2];
	bbox.h = blob[start_idx+3];
	return bbox;
}

template <typename Dtype>
Dtype box_overlap(const Dtype x1, const Dtype w1, const Dtype x2, const Dtype w2){
	Dtype l1 = x1 - w1/2;
	Dtype l2 = x2 - w2/2;
	Dtype left = l1 > l2 ? l1 : l2;
	Dtype r1 = x1 + w1/2;
	Dtype r2 = x2 + w2/2;
	Dtype right = r1 < r2 ? r1 : r2;

	return right - left;
}

template <typename Dtype>
Dtype box_intersection(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	Dtype w = box_overlap(a.x, a.w, b.x, b.w);
	Dtype h = box_overlap(a.y, a.h, b.y, b.h);

	if(w < 0 || h < 0){
		return Dtype(0);
	}

	Dtype area = w * h;
	return area;
}

template <typename Dtype>
Dtype box_union(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	Dtype i = box_intersection(a, b);
	Dtype u = a.w * a.h + b.w * b.h - i;
	return u;
}

template <typename Dtype>
Dtype box_rmse(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2) + pow(a.w - b.w, 2) + pow(a.h-b.h, 2));
}

template <typename Dtype>
Dtype box_iou(const BoundingBox<Dtype>& a, const BoundingBox<Dtype>& b){
	return box_intersection(a, b) / box_union(a, b);
}

template<typename Dtype>
void convert_yolo_detections(const float *predictions, float **probs, YoloBox *boxes,
		const int grid_width, const int boxes_per_cell, const int classes, const bool square,
		const int w, const int h, const bool only_objectness, const Dtype prob_threshold) {

	   for (int i = 0; i < grid_width * grid_width; ++i) {
	       int row = i / grid_width;
	       int col = i % grid_width;
	       for (int n = 0; n < boxes_per_cell; ++n) {
	    	   int index = i * boxes_per_cell + n;
	           int p_index = grid_width * grid_width * classes + i * boxes_per_cell + n;
	           float scale = predictions[p_index];
	           int box_index = grid_width * grid_width * (classes + boxes_per_cell) +
	                (i * boxes_per_cell + n) * 4;
	           boxes[index].x = (predictions[box_index + 0] + col) / grid_width * w;
	           boxes[index].y = (predictions[box_index + 1] + row) / grid_width * h;
	           boxes[index].width = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
	            boxes[index].height = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;
	            for (int j = 0; j < classes; ++j) {
	                int class_index = i * classes;
	                float prob = scale * predictions[class_index + j];
	                probs[index][j] = (prob > prob_threshold) ? prob : 0;
	            }
	            if (only_objectness) {
	                probs[index][0] = scale;
	            }
	       }
	   }
}

template <typename Dtype>
void non_maximum_suppression(const YoloBox *boxes, float **probs, int numBoxes, int classes,
		Dtype iou_threshold) {
	for (int i = 0; i < classes; ++i) {
		bool any = false;
	    for (int k = 0; k < classes; ++k) {
	    	any = any || (probs[i][k] > 0);
	    }
	    if (!any) continue;

	    for (int j = i + 1; j < numBoxes; ++j) {
	    	if (boxes[i].iou(boxes[j]) > iou_threshold) {
	        for (int k = 0; k < classes; ++k) {
	        	if (probs[i][k] < probs[j][k])
	        		probs[i][k] = 0;
	            else
	            	probs[j][k] = 0;
	        }
	    }
	}
}

}

} // namespace caffe

#endif // CAFFE_YOLO_UTILS_HPP_
