#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe{

template<typename Dtype>
Dtype YoloSolver<Dtype>::GetLearningRate()
{
	Dtype rate;
	rate = this->param_.base_lr();
	const string& lr_policy = this->param_.lr_policy();
	if (lr_policy == "multistep"){
		if (this->current_step_ < this->param_.stepvalue_size() &&
		    this->iter_ >= this->param_.stepvalue(this->current_step_)) {
		    this->current_step_++;
		    LOG(INFO) << "MultiStep Status: Iteration " <<
		    this->iter_ << ", step = " << this->current_step_;
		}
		for(int i=0; i<this->param_.stepvalue_size(); ++i){
			if(this->param_.stepvalue(i) > this->iter_){
				return rate;
			}
			rate *= this->param_.scalevalue(i);
		}
	}
	else {
	    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
	  }

	return rate;
}

#ifndef CPU_ONLY
template <typename Dtype>
void yolo_update_bias_gpu(int N, Dtype* data, Dtype* diff, Dtype rate,
    int batch_size, Dtype momentum);

template <typename Dtype>
void yolo_update_weights_gpu(int N, Dtype* data, Dtype* diff, Dtype rate,
    int batch_size, Dtype momentum, Dtype decay);
#endif

template<typename Dtype>
void YoloSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){

	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	Dtype momentum = this->param_.momentum();
	Dtype decay = this->param_.weight_decay();
	Dtype batch_size = this->param_.batch_size();
	Dtype scale_diff = rate / batch_size;
	Dtype scale_weight_data = Dtype(-1) * decay * batch_size;

	switch (Caffe::mode()) {
	  case Caffe::CPU: {

		if(net_params[param_id]->shape().size() == 1){
			// bias_data = 1 * bias_data + scale_diff * bias_diff
   			caffe_cpu_axpby(net_params[param_id]->count(), scale_diff,
			              net_params[param_id]->cpu_diff(), Dtype(1),
						  net_params[param_id]->mutable_cpu_data());

		    // bias_diff = momentum * bias_diff
		    caffe_scal(net_params[param_id]->count(), momentum,
		    		   net_params[param_id]->mutable_cpu_diff());

		}
		else{
			// weights_diff = 1 * weights_diff + scale_weight_data * weights_data
			caffe_cpu_axpby(net_params[param_id]->count(), scale_weight_data,
	    	            	net_params[param_id]->cpu_data(), Dtype(1),
							net_params[param_id]->mutable_cpu_diff());

			// weights_data = 1 * weights_data + scale_diff * weights_diff
			caffe_cpu_axpby(net_params[param_id]->count(), scale_diff,
	    	              	net_params[param_id]->cpu_diff(), Dtype(1),
	    				    net_params[param_id]->mutable_cpu_data());
		}
	    break;
	  }
	  case Caffe::GPU: {
#ifndef CPU_ONLY
		  if(net_params[param_id]->shape().size() == 1){
			  yolo_update_bias_gpu(net_params[param_id]->count(),
					  	  	  	  net_params[param_id]->mutable_gpu_data(),
								  net_params[param_id]->mutable_gpu_diff(),
								  rate, batch_size, momentum);
		  }
		  else{
			  yolo_update_weights_gpu(net_params[param_id]->count(),
					  	  	  	  	  net_params[param_id]->mutable_gpu_data(),
									  net_params[param_id]->mutable_gpu_diff(),
									  rate, batch_size, momentum, decay);
		  }

#else
		  NO_GPU;
#endif
		break;
	  }
	  default:
	    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	  }

}

template<typename Dtype>
void YoloSolver<Dtype>::ApplyUpdate(){
	Dtype rate = GetLearningRate();
	LOG(INFO) << "Current learning rate = " << rate ;
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
	    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
	 }
	 for (int param_id = 0; param_id < this->net_->learnable_params().size();
	       ++param_id) {
	    ComputeUpdateValue(param_id, rate);
	  }
	  //this->net_->Update();

}

INSTANTIATE_CLASS(YoloSolver);
REGISTER_SOLVER_CLASS(Yolo);

}

