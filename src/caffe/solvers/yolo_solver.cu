#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void UpdateWeights_gpu(int N, const Dtype* gpu_data, Dtype* mutable_gpu_data, const Dtype* gpu_diff,
		Dtype* mutable_gpu_diff, Dtype rate, int batch_size, Dtype momentum, Dtype decay){
	caffe_gpu_axpy<Dtype>(N, -decay*batch_size, gpu_data, mutable_gpu_diff); 
    caffe_gpu_axpy<Dtype>(N, rate/batch_size, gpu_diff, mutable_gpu_data);
    caffe_gpu_scal<Dtype>(N, momentum, mutable_gpu_diff);
}

template void UpdateWeights_gpu(int, const float*, float*, const float*, float*, 
		float, int, float, float);
template void UpdateWeights_gpu(int, const double*, double*, const double*, double*, 
		double, int, double, double);
		
template <typename Dtype>
void UpdateBias_gpu(int N, const Dtype* gpu_data, Dtype* mutable_gpu_data, const Dtype* gpu_diff,
		Dtype* mutable_gpu_diff, Dtype rate, int batch_size, Dtype momentum){
    caffe_gpu_axpy<Dtype>(N, rate/batch_size, gpu_diff, mutable_gpu_data);
    caffe_gpu_scal<Dtype>(N, momentum, mutable_gpu_diff);
}

template void UpdateBias_gpu(int, const float*, float*, const float*, float*,
		float, int, float);
template void UpdateBias_gpu(int, const double*, double*, const double*, double*, 
		double, int, double);
		
		
}  // namespace caffe
