#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  K_ = bottom[0]->count() / bottom[0]->num();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";
  (*top)[0]->Reshape(bottom[0]->num(), N_, 1, 1);
  // Set up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Dump(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_data = (*top)[1]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();

  int num_input = bottom[0]->num();
  int dim_input = bottom[0]->count()/num_input;
  int num_output = (*top)[0]->num();
  int dim_output = (*top)[0]->count()/num_output;

  std::ofstream out;
  if (this->layer_param_.inner_product_param().has_dump_weight()) {
    LOG(INFO) << "Dumping weight " << N_ << " " << K_;
    out.open(this->layer_param_.inner_product_param().dump_weight().c_str(),ios::out);
    for (int i=0; i<N_; i++) {
      for (int j=0; j<K_; j++) {
        out << weight[i*K_ + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  if (this->layer_param_.inner_product_param().has_dump_bias()) {
    LOG(INFO) << "Dumping bias " << N_;
    out.open(this->layer_param_.inner_product_param().dump_bias().c_str(),ios::out);
    for (int i=0; i<N_; i++) {
      out << bias[i] << std::endl;
    }
    out.close();
  }

  if (this->layer_param_.inner_product_param().has_dump_input()) {
    LOG(INFO) << "Dumping input " << num_input << " " << dim_input;
    out.open(this->layer_param_.inner_product_param().dump_input().c_str(),ios::out);
    for (int i=0; i<num_input; i++) {
      for (int j=0; j<dim_input; j++) {
        out << bottom_data[i*dim_input + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  if (this->layer_param_.inner_product_param().has_dump_output()) {
    LOG(INFO) << "Dumping output " << num_output << " " << dim_output;
    out.open(this->layer_param_.inner_product_param().dump_output().c_str(),ios::out);
    for (int i=0; i<num_output; i++) {
      for (int j=0; j<dim_output; j++) {
	LOG(INFO) << i << " " << j;
        out << top_data[i*dim_output + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  Dump(bottom, top);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
