#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DumpLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "bottom[0].num() != bottom[1].num()";
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 2);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);

  // actual number of rating in the whole space
  itact_item_ = bottom[2]->num();
  const Dtype* itact_count_ = bottom[2]->cpu_data();
  num_rating_ = itact_count_[(itact_item_-1)*2] + itact_count_[(itact_item_-1)*2+1];
  CHECK_GE(bottom[0]->num(), num_rating_) << "offset exceed end of space";
}

template <typename Dtype>
void DumpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  std::ofstream out;
  if(!this->layer_param_.dump_param().has_dump_file()){
    LOG(FATAL)<<"data dump file do not exist.";
    return;
  }
  out.open(this->layer_param_.dump_param().dump_file().c_str(),ios::out|ios::app);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  for (int i = 0; i < num_rating_; ++i) {
    out << bottom_data[i] << " " << bottom_label[i] << std::endl;
  }
  out.close();
  (*top)[0]->mutable_cpu_data()[0] = num_rating_;
}

INSTANTIATE_CLASS(DumpLayer);

}  // namespace caffe
