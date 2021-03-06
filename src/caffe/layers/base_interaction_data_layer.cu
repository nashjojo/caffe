#include <vector>

#include "caffe/interaction_data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingInteractionDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "Forward_gpu";
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  caffe_copy(prefetch_itact_data_.count(), prefetch_itact_data_.cpu_data(),
             (*top)[2]->mutable_gpu_data());
  caffe_copy(prefetch_itact_label_.count(), prefetch_itact_label_.cpu_data(),
             (*top)[3]->mutable_gpu_data());
  caffe_copy(prefetch_itact_count_.count(), prefetch_itact_count_.cpu_data(),
             (*top)[4]->mutable_gpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_CLASS(BasePrefetchingInteractionDataLayer);

}  // namespace caffe
