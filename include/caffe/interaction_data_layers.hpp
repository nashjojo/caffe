#ifndef CAFFE_INTERACTION_DATA_LAYERS_HPP_
#define CAFFE_INTERACTION_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseInteractionDataLayer : public Layer<Dtype> {
 public:
  explicit BaseInteractionDataLayer(const LayerParameter& param);
  virtual ~BaseInteractionDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingInteractionDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  int datum_channels() const { return datum_channels_; }
  int datum_height() const { return datum_height_; }
  int datum_width() const { return datum_width_; }
  int datum_size() const { return datum_size_; }

 protected:
  TransformationParameter transform_param_;
  DataTransformer<Dtype> data_transformer_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  Blob<Dtype> data_mean_;
  const Dtype* mean_;
  Caffe::Phase phase_;
  bool output_labels_;
};

template <typename Dtype>
class BasePrefetchingInteractionDataLayer :
    public BaseInteractionDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingInteractionDataLayer(const LayerParameter& param)
      : BaseInteractionDataLayer<Dtype>(param) {}
  virtual ~BasePrefetchingInteractionDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
  // used specific for interaction data
  Blob<Dtype> prefetch_itact_data_;
  Blob<Dtype> prefetch_itact_label_;
  Blob<Dtype> prefetch_itact_count_;
};

template <typename Dtype>
class InteractionDataLayer : public BasePrefetchingInteractionDataLayer<Dtype> {
 public:
  explicit InteractionDataLayer(const LayerParameter& param)
      : BasePrefetchingInteractionDataLayer<Dtype>(param) {}
  virtual ~InteractionDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 5; }
  // Only 4 blobs are allowed: data, label, interaction, interaction label

 protected:
  virtual void InternalThreadEntry();

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

}  // namespace caffe

#endif  // CAFFE_INTERACTION_DATA_LAYERS_HPP_
