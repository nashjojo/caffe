#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithFixedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithFixedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithFixedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      loss -= log(std::max(prob_data[i * dim +
          static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
                           Dtype(FLT_MIN)));
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithFixedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "SoftmaxWithFixedLossLayer<Dtype>::Backward_cpu";
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    
    // Make bottom_diff all zeros.
    // caffe_copy(prob_.count(), prob_data, bottom_diff);
    caffe_set(prob_.count(), Dtype(0), bottom_diff);
    int data_dimemsion = (*bottom)[0]->count() / (*bottom)[0]->num();
    CHECK_LT(this->layer_param_.softmax_param().fixed_label(), data_dimemsion) << "fixed_label " << this->layer_param_.softmax_param().fixed_label() << " >= data dimension " << data_dimemsion;
    if (this->layer_param_.softmax_param().fixed_label() >= 0) {
      LOG(INFO) << "SoftmaxWithFixedLossLayer<Dtype>::Backward_cpu fixed_label " << this->layer_param_.softmax_param().fixed_label();
    } else {
      LOG(INFO) << "SoftmaxWithFixedLossLayer<Dtype>::Backward_cpu backpropagate true label";
    }

    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();

    // WARNING: 
    // The system could not detect out of bound memory access. 
    // We have to mamually do that.
    // LOG(INFO) << "prob.height() " << prob_.height() << " prob_.width() " << prob_.width() << " spatial_dim " << spatial_dim;
    // LOG(INFO) << "SoftmaxWithFixedLossLayer<Dtype>::Backward_cpu prob_ length " << num << "*" << dim << "=" << prob_.count() << " bottom_diff.count" << (*bottom)[0]->count();
    // bottom_diff[prob_.count()+1] = -1;

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        if (this->layer_param_.softmax_param().fixed_label() >= 0) {

          // if (i * dim + this->layer_param_.softmax_param().fixed_label()
          //     * spatial_dim + j >= prob_.count()) {
          //   LOG(INFO) << "Index " << i * dim + this->layer_param_.softmax_param().fixed_label()
          //     * spatial_dim + j << " Larger than bounds " << prob_.count();
          // }

          bottom_diff[i * dim + this->layer_param_.softmax_param().fixed_label()
              * spatial_dim + j] = -1;

        } else {
          bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
              * spatial_dim + j] = -1; // # modified
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithFixedLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithFixedLossLayer);


}  // namespace caffe
