#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/interaction_data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
InteractionDataLayer<Dtype>::~InteractionDataLayer<Dtype>() {
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
void InteractionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "DataLayerSetUp";
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
  DatumInteraction datumItract;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datumItract.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datumItract.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  Datum datum = datumItract.datum();
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
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }

  // interaction_data have variable lenth, maybe we just use max length?
  inact_total_size = this->layer_param_.data_param().batch_size() * this->layer_param_.data_param().itact_size();
  // LOG(INFO) << "setting up prefetches_data/label batch_size*itact_size " << this->layer_param_.data_param().batch_size()<<"*"<<this->layer_param_.data_param().itact_size();

  (*top)[2]->Reshape(
      inact_total_size, 2, 1, 1);
  this->prefetch_itact_data_.Reshape(inact_total_size, 2, 1, 1);
  (*top)[3]->Reshape(
      inact_total_size, 1, 1, 1);
  this->prefetch_itact_label_.Reshape(inact_total_size, 1, 1, 1);
  (*top)[4]->Reshape(
      this->layer_param_.data_param().batch_size(), 2, 1, 1);
  this->prefetch_itact_count_.Reshape(this->layer_param_.data_param().batch_size(), 2, 1, 1);

  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void InteractionDataLayer<Dtype>::InternalThreadEntry() {
  // LOG(INFO) << "InternalThreadEntry";
  DatumInteraction datumItract;
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  Dtype* top_data_itact = this->prefetch_itact_data_.mutable_cpu_data();
  Dtype* top_label_itact = this->prefetch_itact_label_.mutable_cpu_data();
  Dtype* top_itact_count = this->prefetch_itact_count_.mutable_cpu_data();
  // use a variable length vector to hold first?
  // Hard to know which part of the click data is belong to which, if not fixed length.
  int itact_offset = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // std::cout << "processing itemid:" << item_id << std::endl; 
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datumItract.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datumItract.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    datum = datumItract.datum();
    // LOG(INFO) << "this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);";
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // adding interaction datatype
    CHECK_EQ(datumItract.userid_size(), datumItract.itemid_size()) << "userid and itemid have different length";
    CHECK_EQ(datumItract.userid_size(), datumItract.rating_size()) << "userid and rating have different length";
    CHECK_GE(this->prefetch_itact_data_.num(), itact_offset+datumItract.userid_size()) << "Max number of rating exceeded!";

    top_itact_count[item_id*2] = itact_offset;
    top_itact_count[item_id*2 + 1] = datumItract.userid_size();
    // LOG(INFO) << "itemid: "<<item_id<<" offset:" << itact_offset << " rating_size: " << datumItract.userid_size();
    // setting interaction data
    for (int itact_id = 0; itact_id < datumItract.userid_size() && itact_offset+itact_id < inact_total_size; ++itact_id) {
      top_data_itact[(itact_offset + itact_id)*2] = datumItract.itemid(itact_id);
      top_data_itact[(itact_offset + itact_id)*2 + 1] = datumItract.userid(itact_id);
      top_label_itact[itact_offset + itact_id] = datumItract.rating(itact_id);
      // LOG(INFO) << "itemid:" << datumItract.itemid(itact_id) << " userid:" << datumItract.userid(itact_id)
      //   << " rating:" << datumItract.rating(itact_id);
    }
    itact_offset += datumItract.userid_size(); // WARNING! ERROR! incase only part of the rating is used. Line 211 ensure this is safe.

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
  // set the extra space in top_data_itact and top_label_itact to 0, in cpu version
  if (itact_offset < this->prefetch_itact_label_.num()) {
    int extra_length = this->prefetch_itact_label_.num() - itact_offset;
    caffe_set(extra_length*2, Dtype(0.), top_data_itact + itact_offset*2);
    caffe_set(extra_length, Dtype(0.), top_label_itact + itact_offset);
  }
}

INSTANTIATE_CLASS(InteractionDataLayer);

}  // namespace caffe
