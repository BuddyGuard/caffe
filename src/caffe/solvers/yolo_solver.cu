#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void YoloUpdateBias(int N, Dtype* gdiff, Dtype* hdata,
	Dtype rate, int batch_size, Dtype momentum) {
  CUDA_KERNEL_LOOP(i, N) {
    gdiff[i] = hdata[i] = hdata[i] + gdiff[i];
    gdiff[i] = gdiff[i] * (-rate/batch_size);
    hdata[i] = momentum * hdata[i];
  }
}

template <typename Dtype>
__global__ void YoloUpdateWeights(int N, Dtype* gdiff, Dtype* hdata, Dtype* gdata,
	Dtype rate, int batch_size, Dtype momentum, Dtype decay) {
  CUDA_KERNEL_LOOP(i, N) {
    gdiff[i] = hdata[i] = (hdata[i] + gdiff[i]) * (-decay * batch_size) * gdata[i];
    gdiff[i] = (-rate/batch_size) * gdiff[i];
    hdata[i] = hdata[i] * momentum;
  }
}

template <typename Dtype>
void yolo_update_bias_gpu(int N, Dtype* gdiff, Dtype* hdata, 
	Dtype rate, int batch_size, Dtype momentum) {
  YoloUpdateBias<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, gdiff, hdata, rate, batch_size, momentum);
  CUDA_POST_KERNEL_CHECK;
}
template void yolo_update_bias_gpu<float>(int , float*, float*, float, int, float);
template void yolo_update_bias_gpu<double>(int, double*, double*, double, int, double);

template <typename Dtype>
void yolo_update_weights_gpu(int N, Dtype* gdiff, Dtype* hdata, Dtype* gdata,
	Dtype rate, int batch_size, Dtype momentum, Dtype decay) {
  YoloUpdateWeights<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, gdiff, hdata, gdata, rate, batch_size, momentum, decay);
  CUDA_POST_KERNEL_CHECK;
}
template void yolo_update_weights_gpu<float>(int , float*, float*, float*, float, int, float, float);
template void yolo_update_weights_gpu<double>(int, double*, double*, double*, double, int, double, double);

}  // namespace caffe
