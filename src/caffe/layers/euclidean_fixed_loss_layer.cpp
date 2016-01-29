#include <vector>
#include <iostream> // for debug

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanWithFixedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EuclideanWithFixedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();

  // LOG(INFO) << "data";
  // const Dtype* data = bottom[0]->cpu_data();
  // for (int i = 0; i < 12; i++){
  //   std::cout << data[i] << "\t";
  // }
  // std::cout << std::endl;
  // LOG(INFO) << "label";
  // const Dtype* label = bottom[1]->cpu_data();
  // for (int i = 0; i < 12; i++){
  //   std::cout << label[i] << "\t";
  // }
  // std::cout << std::endl;

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  // LOG(INFO) << "diff";
  // const Dtype* diff_cpu = diff_.cpu_data();
  // for (int i = 0; i < 12; i++){
  //   std::cout << diff_cpu[i] << "\t";
  // }
  // std::cout << std::endl;

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());

  // std::cout << "dot:" << dot << std::endl;

  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  // Dtype loss = sqrt(dot / bottom[0]->num()); // rmse, temp for movielens.
  (*top)[0]->mutable_cpu_data()[0] = loss;
  // LOG(INFO) << "loss "<< loss;
}

template <typename Dtype>
void EuclideanWithFixedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  Dtype* diff_mutable = diff_.mutable_cpu_data();
  int data_dimemsion = (*bottom)[0]->count() / (*bottom)[0]->num();
  CHECK_LT(this->layer_param_.softmax_param().fixed_label(), data_dimemsion) << "fixed_label " << this->layer_param_.softmax_param().fixed_label() << " >= data dimension " << data_dimemsion;
  if (this->layer_param_.softmax_param().fixed_label() >= 0) {
    LOG(INFO) << "EuclideanWithFixedLossLayer<Dtype>::Backward_cpu fixed_label " << this->layer_param_.softmax_param().fixed_label();

    caffe_set(diff_.count(), Dtype(0), diff_mutable); // set diff to 0
    int num = diff_.num();
    int dim = diff_.count() / num;
    for (int i = 0; i < num; ++i) {
      diff_mutable[i * dim + this->layer_param_.softmax_param().fixed_label()] = 1; // target dimension set to fixed label
    }
  } else {
    LOG(INFO) << "EuclideanWithFixedLossLayer<Dtype>::Backward_cpu backpropagate true label";
  }

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();  // top[0]->cpu_diff()[0] is set to 1 by net that's the total loss of all instance.
      caffe_cpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_cpu_diff());  // b

      
      // const Dtype* diff_cpu = (*bottom)[i]->cpu_diff();
      // std::cout << "diff in loss" << std::endl;
      // for (int i = 0; i < 12; i++){
      //   std::cout << diff_cpu[i] << "\t";
      // }
      // std::cout << std::endl;
    } // ~ propagate_down[i]
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanWithFixedLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanWithFixedLossLayer);

}  // namespace caffe
