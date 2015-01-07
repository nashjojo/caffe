#include <vector>
#include <iostream> // for debug

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RmseLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void RmseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  num_rating_ = bottom[2]->cpu_data()[0];
  int count = bottom[0]->count();
  CHECK_LE(num_rating_, count) << "assigned rating length exceed boundary.";

  // const Dtype* data = bottom[0]->cpu_data();
  // for (int i = 0; i < 10; i++){
  //   std::cout << data[i] << "\t";
  // }
  // std::cout << std::endl;
  // const Dtype* label = bottom[1]->cpu_data();
  // for (int i = 0; i < 10; i++){
  //   std::cout << label[i] << "\t";
  // }
  // std::cout << std::endl;

  caffe_sub(
      num_rating_,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  // const Dtype* diff_cpu = diff_.cpu_data();
  // for (int i = 0; i < 10; i++){
  //   std::cout << diff_cpu[i] << "\t";
  // }
  // std::cout << std::endl;

  Dtype dot = caffe_cpu_dot(num_rating_, diff_.cpu_data(), diff_.cpu_data());

  // std::cout << "dot:" << dot << std::endl;

  // Dtype loss = dot / bottom[0]->num() / Dtype(2);
  Dtype loss = sqrt(dot / num_rating_); // rmse, temp for movielens.
  (*top)[0]->mutable_cpu_data()[0] = loss;
  // LOG(INFO) << "loss:" << loss << " num_rating_:" << num_rating_;
}

template <typename Dtype>
void RmseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_rating_;  // top[0]->cpu_diff()[0] is set to 1 by net that's the total loss of all instance.
      caffe_cpu_axpby(
          num_rating_,                        // num_rating_
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_cpu_diff());  // b

      
      // const Dtype* diff_cpu = (*bottom)[i]->cpu_diff();
      // std::cout << "diff in loss" << std::endl;
      // for (int i = 0; i < 10; i++){
      //   std::cout << diff_cpu[i] << "\t";
      // }
      // std::cout << std::endl;
    } // ~ propagate_down[i]
  }
}

#ifdef CPU_ONLY
STUB_GPU(RmseLossLayer);
#endif

INSTANTIATE_CLASS(RmseLossLayer);

}  // namespace caffe
