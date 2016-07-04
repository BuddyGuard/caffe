#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe{

template<typename Dtype>
void YoloSolver<Dtype>::ResetMomentum(){
	if(momentum_ == 0) return;
	rate_ = 0;
	momentum_ = 0;
	decay_ = 0;
#ifndef CPU_ONLY
	UpdateParams();
	momentum_ = this->param_.momentum();
	decay_ = this->param_.weight_decay();
#endif

}

template<typename Dtype>
void YoloSolver<Dtype>::PreSolve(){
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	this->history_.clear();
	for(int i = 0; i < net_params.size(); ++i){
		const vector<int>& shape = net_params[i]->shape();
		this->history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
	}

}

template<typename Dtype>
Dtype YoloSolver<Dtype>::GetLearningRate()
{
	Dtype rate = rate_;
	const string& lr_policy = this->param_.lr_policy();
	if (lr_policy == "multistep"){
		if (this->current_step_ < this->param_.stepvalue_size() &&
		    this->iter_ >= this->param_.stepvalue(this->current_step_)) {
		    this->current_step_++;
		    LOG(INFO) << "MultiStep Status: Iteration " <<
		    this->iter_ << ", step = " << this->current_step_;
		}
		for(int i=0; i < this->param_.stepvalue_size(); ++i){
			if(this->param_.stepvalue(i) > this->iter_){
				return rate;
			}
			rate *= this->param_.scalevalue(i);
			if(this->param_.stepvalue(i) > this->iter_ - 1){
				ResetMomentum();
			}
		}
	}
	else {
	    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
	  }

	return rate;
}

#ifndef CPU_ONLY
template <typename Dtype>
void yolo_update_bias_gpu(int N, Dtype* gdiff, Dtype* hdata,
		Dtype rate, int batch_size, Dtype momentum);

template <typename Dtype>
void yolo_update_weights_gpu(int N, Dtype* gdiff, Dtype* hdata, Dtype* gdata,
		Dtype rate, int batch_size, Dtype momentum, Dtype decay);
#endif

template<typename Dtype>
void YoloSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate, Dtype momentum, Dtype decay){

	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	Dtype batch_size = this->param_.batch_size();
	Dtype scale_diff = Dtype(-rate) / batch_size;
	Dtype scale_weight_data = Dtype(-1) * decay * batch_size;

	switch (Caffe::mode()) {
	  case Caffe::CPU: {

		if(net_params[param_id]->shape().size() == 1){

			caffe_axpy(net_params[param_id]->count(), Dtype(1),
					   net_params[param_id]->cpu_diff(),
					   this->history_[param_id]->mutable_cpu_data());

			caffe_copy(net_params[param_id]->count(),
					   this->history_[param_id]->cpu_data(),
					   net_params[param_id]->mutable_cpu_diff());

			caffe_scal(net_params[param_id]->count(), scale_diff,
					net_params[param_id]->mutable_cpu_diff());

			// bias_data = 1 * bias_data + scale_diff * bias_updates
   			//caffe_axpy(net_params[param_id]->count(), scale_diff,
			//              	net_params[param_id]->cpu_diff(),
			//				net_params[param_id]->mutable_cpu_data());

		    // bias_update = bias_update + momentum * bias_diff
   			caffe_scal(this->history_[param_id]->count(), momentum,
   					this->history_[param_id]->mutable_cpu_data());

		}
		else{

			caffe_axpy(net_params[param_id]->count(), Dtype(1),
					net_params[param_id]->cpu_diff(),
					this->history_[param_id]->mutable_cpu_data());

			// weights_update = 1 * weights_update + scale_weight_data * weights_data
			caffe_axpy(net_params[param_id]->count(), scale_weight_data,
					this->history_[param_id]->cpu_data(),
					this->history_[param_id]->mutable_cpu_data());

			caffe_copy(net_params[param_id]->count(),
					this->history_[param_id]->cpu_data(),
					net_params[param_id]->mutable_cpu_diff());

			caffe_scal(net_params[param_id]->count(), scale_diff,
					net_params[param_id]->mutable_cpu_diff());

			// weights_data = 1 * weights_data + scale_diff * weights_diff
			//caffe_axpy(net_params[param_id]->count(), scale_diff,
	    	//              	net_params[param_id]->cpu_data(),
	    	//			    net_params[param_id]->mutable_cpu_data());

			// weights_update = momentum * weights_update
			caffe_scal(this->history_[param_id]->count(), momentum,
					this->history_[param_id]->mutable_cpu_diff());

		}
	    break;
	  }
	  case Caffe::GPU: {
#ifndef CPU_ONLY
		  if(net_params[param_id]->shape().size() == 1){
			  yolo_update_bias_gpu(net_params[param_id]->count(),
					  	  	  	  net_params[param_id]->mutable_gpu_diff(),
								  this->history_[param_id]->mutable_gpu_data(),
								  rate, batch_size, momentum);
		  }
		  else{
			  yolo_update_weights_gpu(net_params[param_id]->count(),
					  	  	  	  	  net_params[param_id]->mutable_gpu_diff(),
									  this->history_[param_id]->mutable_gpu_data(),
									  net_params[param_id]->mutable_gpu_data(),
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
void YoloSolver<Dtype>::UpdateParams(){
	Dtype rate = GetLearningRate();
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		   LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
	}
	for (int param_id = 0; param_id < this->net_->learnable_params().size();
		     ++param_id) {
		ComputeUpdateValue(param_id, rate, momentum_, decay_);
	}
	this->net_->Update();
}

template<typename Dtype>
void YoloSolver<Dtype>::ApplyUpdate(){
	rate_ = this->param_.base_lr();
	momentum_ = this->param_.momentum();
	decay_ = this->param_.weight_decay();
	UpdateParams();
	if(avg_loss_ == 0){
			avg_loss_ = this->smoothed_loss_;
	}

	avg_loss_ = 0.9 * avg_loss_ + 0.1 *  this->smoothed_loss_;

	LOG(INFO) << this->iter_ <<": "<<this->smoothed_loss_<< ", "<<avg_loss_<<" avg, "<<GetLearningRate()<<" rate";

}

INSTANTIATE_CLASS(YoloSolver);
REGISTER_SOLVER_CLASS(Yolo);

}

