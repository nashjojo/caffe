#include <string>
#include <vector>

#include "caffe/interaction_data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseInteractionDataLayer<Dtype>::BaseInteractionDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseInteractionDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  output_labels_ = true;
  DataLayerSetUp(bottom, top);
  // The subclasses should setup the datum channels, height and width
  CHECK_GT(datum_channels_, 0);
  CHECK_GT(datum_height_, 0);
  CHECK_GT(datum_width_, 0);
  if (transform_param_.crop_size() > 0) {
    CHECK_GE(datum_height_, transform_param_.crop_size());
    CHECK_GE(datum_width_, transform_param_.crop_size());
  }
  // check if we want to have mean
  if (transform_param_.has_mean_file()) {
    const string& mean_file = transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_GE(data_mean_.num(), 1);
    CHECK_GE(data_mean_.channels(), datum_channels_);
    CHECK_GE(data_mean_.height(), datum_height_);
    CHECK_GE(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  mean_ = data_mean_.cpu_data();
  data_transformer_.InitRand();
}

template <typename Dtype>
void BasePrefetchingInteractionDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseInteractionDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  // this haves a variable length, how? Use maximum length?
  this->prefetch_itact_data_.mutable_cpu_data();
  this->prefetch_itact_label_.mutable_cpu_data();
  this->prefetch_itact_count_.mutable_cpu_data();

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingInteractionDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingInteractionDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingInteractionDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "Forward_cpu";
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  caffe_copy(prefetch_itact_data_.count(), prefetch_itact_data_.cpu_data(),
             (*top)[2]->mutable_cpu_data());
  caffe_copy(prefetch_itact_label_.count(), prefetch_itact_label_.cpu_data(),
             (*top)[3]->mutable_cpu_data());
  caffe_copy(prefetch_itact_count_.count(), prefetch_itact_count_.cpu_data(),
             (*top)[4]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingInteractionDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseInteractionDataLayer);
INSTANTIATE_CLASS(BasePrefetchingInteractionDataLayer);

}  // namespace caffe
