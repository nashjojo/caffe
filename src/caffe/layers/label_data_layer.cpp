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
LabelDataLayer<Dtype>::~LabelDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void LabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

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

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label allocation 
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), this->label_dim_, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        this->label_dim_, 1, 1);
    (*top)[2]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_id_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void LabelDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_id = NULL;
  const Dtype* mem_label = this->label_set_.cpu_data();
  int memID = 0, item_origin_ID = 0;
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
    top_id = this->prefetch_id_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      // read hash map and copy
      // top_label[item_id] = datum.label();
      item_origin_ID = datum.label();

      // TESTING
      // item_origin_ID = item_origin_ID%2;

      // check if key exists
      if (this->ID2Idx_.find(item_origin_ID) == this->ID2Idx_.end()) {
        LOG(FATAL) << "item_origin_ID " << item_origin_ID << " not found";
      } else {
        memID = this->ID2Idx_[item_origin_ID];
      }      
      caffe_copy(this->label_dim_, mem_label + memID*this->label_dim_,
             top_label + item_id*this->label_dim_);
      
      top_id[item_id] = item_origin_ID;
      // LOG(INFO) << "item_origin_ID " << item_origin_ID << " memID " << memID;
    }

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(LabelDataLayer);

}  // namespace caffe
