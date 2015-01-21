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
void SnapshotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  vector<Blob<Dtype>*>* top) {
  if(!this->layer_param_.snapshot_param().has_dump_file()){
    LOG(FATAL)<<"data dump file do not exist.";
    return;
  }
  out.open(this->layer_param_.snapshot_param().dump_file().c_str(),ios::out|ios::app);
  LOG(INFO)<< "Dumping to " << this->layer_param_.snapshot_param().dump_file();
}

template <typename Dtype>
SnapshotLayer<Dtype>::~SnapshotLayer() {
  LOG(INFO)<< "Closing file " << this->layer_param_.snapshot_param().dump_file();
  out.close();
}

template <typename Dtype>
void SnapshotLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  (*top)[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SnapshotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count()/bottom[0]->num();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i];
  }
  // dumping to file
  out << "forward" << std::endl;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      out << bottom_data[i*dim + j] << " ";
    }
    out << std::endl;
  }
}

template <typename Dtype>
void SnapshotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const int dim = (*bottom)[0]->count()/(*bottom)[0]->num();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }
    // dumping to file
    out << "backward" << std::endl;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        out << top_diff[i*dim + j] << " ";
      }
      out << std::endl;
    }
  }
}

INSTANTIATE_CLASS(SnapshotLayer);

}  // namespace caffe
