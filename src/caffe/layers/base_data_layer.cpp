#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
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

  // for visualization
  instance_so_far_ = 0;
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // std::cout << "BasePrefetchingDataLayer Datalayer Forward_cpu" << std::endl;
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  std::cout << "BasePrefetchingDataLayer Datalayer Backward_cpu" << std::endl;
  std::cout << this->layer_param_.data_param().data_dump() << std::endl;

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  // const int* id = reinterpret_cast<const int*>(top[1]->cpu_data());
  const Dtype* id = top[1]->cpu_data();
  // top[1]->cpu_data() has become all 0s.

  const Dtype scale = this->layer_param_.transform_param().scale();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  int channels = this->datum_channels_;
  string dump_path = this->layer_param_.data_param().data_dump();
  int num = top[0]->num();
  int dim = top[0]->count() / num;

  char filename[256];
  int img_offset = 0;
  int channel_offset = 0;
  Dtype value_in = 0;
  uint8_t value_out = 0;
  Dtype max_val = 0;
  Dtype min_val = 0;

  // for Caffe::Test transformed type
  int batch_size = this->layer_param_.data_param().batch_size();
  CHECK_EQ(num, batch_size) << "Test batch size do not match";

  // std::cout << "num is " << num << std::endl;
  // std::cout << "dim is " << dim << std::endl;

  for (int itemid = 0; itemid < num; ++itemid ) {
    // std::cout << "itemid is " << itemid << " " << id[itemid] << std::endl;

    if (this->layer_param_.data_param().visualize()>=2) {
      std::ofstream outfile;
      sprintf( filename, "%s/saliency/%.0f.txt", dump_path.c_str(), id[itemid] ); // itemid+this->instance_so_far_
      outfile.open(filename);
      for (int h = 0; h < crop_size; ++h) {
        for (int w = 0; w < crop_size; ++w) {
          max_val = 0;
          for (int c = 0; c < channels; ++c) {
            img_offset = itemid*channels*crop_size*crop_size;
            channel_offset = c*crop_size*crop_size;
            value_in = top_diff[img_offset + channel_offset + h*crop_size + w ] / scale;
            if (abs(value_in) > max_val) {
              max_val = abs(value_in);
            }
            // if(value_in > 0) {
            //  LOG(INFO)<<"itemid "<<itemid <<" channel:"<<c <<" height:"<<h <<" width:"<<w <<" value_in:"<<value_in <<" value_out:"<<value_out;
            // }
          }
          outfile << max_val << "\t";
        }
        outfile << std::endl;
      }
      // save image files
      outfile.close();
    }

    if (this->layer_param_.data_param().visualize()==1 || this->layer_param_.data_param().visualize()==3) {
      // data_diff -> image
      max_val = 0;
      min_val = 255;
      cv::Mat original_img = cv::Mat::zeros(crop_size,crop_size, CV_8UC3);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            img_offset = itemid*channels*crop_size*crop_size;
            channel_offset = c*crop_size*crop_size;

            value_in = top_data[img_offset + channel_offset + h*crop_size + w ];
            value_out = static_cast<uint8_t>( value_in*1.0 / scale + this->mean_[channel_offset + h*crop_size + w] );
            // std::cout << c << " " << h << " " << w << " " << static_cast<int>(value_out) << std::endl;
            original_img.at<cv::Vec3b>(h,w)[c] = value_out;
            //if(value_in > 0) {
              //LOG(INFO)<<"itemid "<<itemid <<" channel:"<<c <<" height:"<<h <<" width:"<<w <<" value_in:"<<value_in <<" value_out:"<<value_out;
            //}

            // Checking saturation reason
            // if (this->mean_[channel_offset + h*crop_size + w] > max_val) {
            //   max_val = this->mean_[channel_offset + h*crop_size + w];
            // }
            // if (this->mean_[channel_offset + h*crop_size + w] < min_val) {
            //   min_val = this->mean_[channel_offset + h*crop_size + w];
            // }
          }
        }
      }
      // std::cout << "itemid is " << itemid << " " << id[itemid] << " max_value " << max_val << " min_value " << min_val << std::endl;
      // save image files
      // sprintf( filename, "%s/original/%d.png", dump_path.c_str(), itemid );
      sprintf( filename, "%s/original/%.0f.png", dump_path.c_str(), id[itemid] ); // itemid+this->instance_so_far_
      cv::imwrite( filename, original_img );
    }
  }
  this->instance_so_far_ = this->instance_so_far_ + num;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
