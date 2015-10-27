#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <hash_map>

namespace caffe {

template <typename Dtype>
void MemoryMappingDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // TODO: Read in label file. 
  // __gnu_cxx::hash_map<int, int> ID2Idx;
  // read in file from 
  this->label_dim_= this->layer_param_.data_param().label_dim();
  this->total_size_= this->layer_param_.data_param().total_size();
  this->label_set_.Reshape(this->total_size_, this->label_dim_, 1, 1);

  LOG(INFO) << "Reading Label from file";
  LOG(INFO) << this->layer_param_.data_param().label_source();
  // LOG(INFO) << this->layer_param_.data_param().label_dim();
  // LOG(INFO) << this->layer_param_.data_param().total_size();
  Dtype* mem_label = NULL;
  mem_label = this->label_set_.mutable_cpu_data();
  Dtype temp = 0;
  int line_id, col_id;
  line_id = 0, col_id = 0;
  std::ifstream infile( this->layer_param_.data_param().label_source().c_str() );
  while (infile) {
    string s;
    if (!getline( infile, s ))
      break;
    CHECK_GE(this->total_size_, line_id+1) << "Label line " << line_id << " exceed maximum label size " << this->total_size_;
    std::istringstream ss( s );
    col_id = 0;
    while (ss) {
      string s;
      if (!getline( ss, s, ',' )) 
        break;
      std::istringstream sss( s );
      sss >> temp;
      // check #dimension consistent
      CHECK_GE(this->label_dim_, col_id+1) << "Label idx " << col_id << " exceed maximum dimension " << this->label_dim_;
      // std::cout << temp << " ";
      // std::cout << line_id << "," << col_id << ":" << temp << std::endl;
      mem_label[line_id*this->label_dim_ + col_id] = temp;
      col_id ++;
    }
    line_id ++;
    // std::cout << std::endl;
  }
  int label_vector = line_id;  

  // Load all <ID>
  LOG(INFO) << this->layer_param_.data_param().label_id();
  int temp_id = 0;
  line_id = 0;
  std::ifstream infile2( this->layer_param_.data_param().label_id().c_str() );
  while (infile2) {
    string s;
    if (!getline( infile2, s ))
      break;
    std::istringstream ss( s );
    ss >> temp_id;
    this->ID2Idx_[temp_id] = line_id;
    // std::cout << temp_id << "," << line_id << std::endl;
    line_id ++;
  }

  // check #ID = #Label_vector
  CHECK_EQ(line_id, label_vector)
      << "#ID and #label dismatch! #ID=" << line_id << " but  #label_vector=" << label_vector;
  LOG(INFO) << "Label Loading Completed";

  // Label Only
  (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(), this->label_dim_, 1, 1);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MemoryMappingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_id = bottom[0]->cpu_data();
  Dtype* top_label = (*top)[0]->mutable_cpu_data();  
  const Dtype* mem_label = this->label_set_.cpu_data();
  
  int memID = 0, item_origin_ID = 0;
  const int batch_size = this->layer_param_.data_param().batch_size();
  
  for (int item_id = 0; item_id < batch_size; ++item_id) { 
    // read hash map and copy
    item_origin_ID = bottom_id[item_id];
    
    // TESTING
    // item_origin_ID = item_origin_ID%2;

   // check if key exists
    if (this->ID2Idx_.find(item_origin_ID) == this->ID2Idx_.end()) {
      LOG(FATAL) << "item_origin_ID " << item_origin_ID << " not found";
    } else {
      memID = this->ID2Idx_[item_origin_ID];
    }      
    caffe_copy(this->label_dim_, mem_label + memID*this->label_dim_, top_label + item_id*this->label_dim_);
    // LOG(INFO) << "item_origin_ID " << item_origin_ID << " memID " << memID;
    // for (int i = 0; i < this->label_dim_; ++i) {
    //   std::cout << mem_label[memID*this->label_dim_+i] << " ";
    // }
    // std::cout << std::endl;
  } 
}

INSTANTIATE_CLASS(MemoryMappingDataLayer);

}  // namespace caffe
