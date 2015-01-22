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
void DumpFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  vector<Blob<Dtype>*>* top) {
  if(!this->layer_param_.dump_param().has_dump_file()){
    LOG(FATAL)<<"data dump file do not exist.";
    return;
  }
  out.open(this->layer_param_.dump_param().dump_file().c_str(),ios::out|ios::app);
  LOG(INFO)<< "Dumping to " << this->layer_param_.dump_param().dump_file();
}

template <typename Dtype>
DumpFeatureLayer<Dtype>::~DumpFeatureLayer() {
  LOG(INFO)<< "Closing file " << this->layer_param_.dump_param().dump_file();
  out.close();
}

template <typename Dtype>
void DumpFeatureLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void DumpFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* itact_data_ = bottom[1]->cpu_data();
  const Dtype* itact_count_ = bottom[2]->cpu_data();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count/num;

  int item_offset = 0, item_real_id = 0;
  // dumping features
  for (int itemid = 0; itemid < num; ++itemid) {
    item_offset = itact_count_[itemid*2];
    item_real_id = itact_data_[item_offset*2];
    out << item_real_id << "," << 0;
    for (int j = 0; j < dim; ++j) {
      out << "," << bottom_data[itemid*dim + j] ;
    }
    out << std::endl;
  }

  (*top)[0]->mutable_cpu_data()[0] = num;
}

INSTANTIATE_CLASS(DumpFeatureLayer);

}  // namespace caffe
