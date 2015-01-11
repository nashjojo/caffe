#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RmseLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  num_rating_ = bottom[2]->cpu_data()[0];
  int count = bottom[0]->count();
  CHECK_LE(num_rating_, count) << "assigned rating length exceed boundary.";
  caffe_gpu_sub(
      num_rating_,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  if (bias_!=0) {
    caffe_gpu_add_scalar(num_rating_, bias_, diff_.mutable_gpu_data());
  }
  Dtype dot;
  caffe_gpu_dot(num_rating_, diff_.gpu_data(), diff_.gpu_data(), &dot);
  // Dtype loss = dot / bottom[0]->num() / Dtype(2);
  Dtype loss = sqrt(dot / num_rating_); // rmse, temp for movielens.
  (*top)[0]->mutable_cpu_data()[0] = loss;
  // LOG(INFO) << "loss:" << loss << " num_rating_:" << num_rating_;
}

template <typename Dtype>
void RmseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      // LOG(INFO) << "propagate_down[i]:" << i << " top[0]->cpu_diff()[0]:" << top[0]->cpu_diff()[0]; // top[0]->cpu_diff()[0] is set to 1 by net that's the total loss of all instance.
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_rating_;
      caffe_gpu_axpby(
          num_rating_,                        // actual number of rating
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(RmseLossLayer);

}  // namespace caffe
