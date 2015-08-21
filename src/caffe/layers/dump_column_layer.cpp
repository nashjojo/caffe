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
void DumpColumnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  vector<Blob<Dtype>*>* top) {
  if(!this->layer_param_.dump_param().has_dump_file()){
    LOG(FATAL)<<"data dump file do not exist.";
    return;
  }
  out.open(this->layer_param_.dump_param().dump_file().c_str(),ios::out|ios::app);
  LOG(INFO)<< "Dumping to " << this->layer_param_.dump_param().dump_file();
}

template <typename Dtype>
DumpColumnLayer<Dtype>::~DumpColumnLayer() {
  LOG(INFO)<< "Closing file " << this->layer_param_.dump_param().dump_file();
  out.close();
}

template <typename Dtype>
void DumpColumnLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void DumpColumnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count()/bottom[0]->num();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      out << bottom_data[i*dim + j] << " ";
    }
    out << std::endl;
  }

  (*top)[0]->mutable_cpu_data()[0] = bottom[0]->num();
}

INSTANTIATE_CLASS(DumpColumnLayer);

}  // namespace caffe
