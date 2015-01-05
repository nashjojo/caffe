#include <vector>
#include <iostream> // add for debug

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/*  find input latent factor dimension
    max_number of interaction size per 
    read user size config/item size config
    set up V
    set up bias 
    * Called one time. Maybe called by dummy blobs. Do not read blobs contents.
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "LayerSetUp";
  force_cpu_ = this->layer_param_.matrix_fact_param().force_cpu();
  bias_term_ = this->layer_param_.matrix_fact_param().bias_term();
  num_user_ = this->layer_param_.matrix_fact_param().num_user();
  num_item_ = this->layer_param_.matrix_fact_param().num_item();
  img_weight_ = this->layer_param_.matrix_fact_param().img_weight();
  feature_weight_ = this->layer_param_.matrix_fact_param().feature_weight();
  num_latent_ = bottom[0]->count() / bottom[0]->num();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Intialize the user latent feature
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, num_user_, num_latent_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.matrix_fact_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Intialize the item latent feature
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, num_item_, num_latent_));
    // fill the weights
    weight_filler->Fill(this->blobs_[1].get());
    // If necessary, intiialize and fill the global bias term
    if (bias_term_) {
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.matrix_fact_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->param_propagate_down_[0] = true;  // back propogate U
  this->param_propagate_down_[1] = true;  // back propogate v
  this->param_propagate_down_[2] = true;  // back propogate global bias

  itact_size_ = this->layer_param_.matrix_fact_param().itact_size();
  itact_item_ = bottom[0]->num(); // number of items/ batch size, batch size may vary!
  max_rating_size_ = bottom[1]->num(); // MAX input number of rating
  // temp space filtering
  this->user_feature_buffer_.Reshape(1, 1, itact_size_, num_latent_);
  this->item_feature_buffer_.Reshape(1, 1, itact_item_, num_latent_);
  this->item_feature_mixed_.Reshape(1, 1, itact_item_, num_latent_);
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, max_rating_size_);
    caffe_set(max_rating_size_, Dtype(1.), bias_multiplier_.mutable_cpu_data());
  }

  // // debug: setting user feature
  // std::cout << "debug: setting user features" << std::endl;
  // Dtype* user_feature = this->blobs_[0]->mutable_cpu_data();
  // for (int i = 0; i < num_user_; i++ ) {
  //   for (int j = 0; j < num_latent_; j++) {
  //     user_feature[i*num_latent_ + j] = Dtype(i);
  //   }
  // }
}

/*  
  build a hashmap for each item and each user
  * called every time before the forward, doing bottom blob dependent allocation. 
  Assume blob's content is good. 
  Output prediction with length of max rating number. (equal to input rating length)
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "MatrixFactorizeLayer<Dtype>::Reshape";
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), num_latent_) << "Input item latent factor dimension "
    "incompatible with inner user latent factor dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num()) << "Input item number not equal to "
    "offset number.";
  
  (*top)[0]->Reshape(bottom[1]->num(), 1, 1, 1);
  // Set up the bias multiplier
  if (max_rating_size_ != bottom[1]->num() && bias_term_) {
    max_rating_size_ = bottom[1]->num();
    bias_multiplier_.Reshape(1, 1, 1, max_rating_size_);
    caffe_set(max_rating_size_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  // if necessary, chagned the size of user_latent and item latent
  if (itact_size_ != this->layer_param_.matrix_fact_param().itact_size()) {
    itact_size_ = this->layer_param_.matrix_fact_param().itact_size();
    this->user_feature_buffer_.Reshape(1, 1, itact_size_, num_latent_);
    LOG(INFO) << "Changed itact_size_ to " << itact_size_;
  }
  if (itact_item_ != bottom[0]->num()) {
    itact_item_ = bottom[0]->num(); // number of items/ batch size, batch size may change!
    this->item_feature_buffer_.Reshape(1, 1, itact_item_, num_latent_);
    this->item_feature_mixed_.Reshape(1, 1, itact_item_, num_latent_);
    LOG(INFO) << "Changed itact_item_ to " << itact_item_;
  }

  // actual number of rating in the whole space
  const Dtype* itact_count_ = bottom[2]->cpu_data();
  num_rating_ = itact_count_[(itact_item_-1)*2] + itact_count_[(itact_item_-1)*2+1];

  // clear computing flags each round.
  gen_item_diff_ = false;

  Build_map(bottom);

  // // debug: setting item features
  // std::cout << "debug: setting item features" << std::endl;
  // Dtype* item_feature = bottom[0]->mutable_cpu_data();
  // for (int i = 0; i < itact_item_; i++ ) {
  //   for (int j = 0; j < num_latent_; j++) {
  //     item_feature[i*num_latent_ + j] = Dtype(i);
  //   }
  // }
}

// building rating map user2itemid and itemid2relative_id
// need to clear first
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Build_map(const vector<Blob<Dtype>*>& bottom) {
  // LOG(INFO) << "Build_map";
  // Setup map for each batch, in cpu model this may be slow.
  const Dtype* itact_data_ = bottom[1]->cpu_data();
  const Dtype* itact_count_ = bottom[2]->cpu_data();

  // LOG(INFO) << "map cleared";
  itemid2relative_id.clear();
  user2itemid.clear();

  int item_offset = 0, rating_size = 0;
  int item_real_id = 0, userid = 0, rating_idx = 0;
  // relative itemid
  for (int itemid = 0; itemid < itact_item_; ++itemid ) {
    item_offset = itact_count_[itemid*2];
    rating_size = itact_count_[itemid*2+1];
    item_real_id = itact_data_[item_offset*2];
    itemid2relative_id[item_real_id] = itemid; // map real itemid to relative itemid

    // LOG(INFO) << "itemid:" << itemid << " item_real_id:" << item_real_id << " offset:" << item_offset << " rating_size: " << rating_size;

    for (int rating_cnt = 0; rating_cnt < rating_size; ++rating_cnt) {
      rating_idx = item_offset + rating_cnt;
      userid = static_cast<int>(itact_data_[rating_idx*2+1]);
      // LOG(INFO) << "itemid:" << itemid << " item_real_id:" << item_real_id << " userid:" << userid;
      // check if userid exist
      if (user2itemid.find(userid) == user2itemid.end()) {
        // LOG(INFO) << "new userid " << userid << " with rating_idx " << rating_idx;
        vector<int> temp;
        temp.push_back(rating_idx);
        user2itemid[userid] = temp;
      } else {
        vector<int>& temp = user2itemid[userid];
        // LOG(INFO) << "existing userid " << userid << " with " << temp.size() << " ratings.";
        // for (int i = 0; i < temp.size(); i++) {
        //   LOG(INFO) << i << " " << temp[i];
        // }
        temp.push_back(rating_idx); // duplicated <itemid,userid> will not cause error.
        user2itemid[userid] = temp;
      }
    }
  }

  // LOG(INFO) << "Checking user2itemid";
  // // check user2itemid
  // int itemid = 0;
  // map<int, vector<int> >::iterator iter;
  // for (iter=user2itemid.begin(); iter!=user2itemid.end(); ++iter) {
  //   userid = iter->first;
  //   vector<int>& ratingset = iter->second;
  //   LOG(INFO) << "Userid:" << userid << " ratingset_size:" << ratingset.size();
  //   for (int j = 0; j < ratingset.size(); ++j) {
  //     rating_idx = ratingset[j];
  //     item_real_id = static_cast<int>(itact_data_[rating_idx*2]);
  //     itemid = itemid2relative_id[item_real_id];
  //     LOG(INFO) << "itemid_real:" << item_real_id << " itemid:" << itemid;
  //   }
  // }
}

/*  
    for each item,
      fill user feature buffer
      pred in block wise manner
    Add global bias if necessary
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "Forward_cpu";
  const Dtype* user_feature = this->blobs_[0]->cpu_data();
  const Dtype* item_feature = this->blobs_[1]->cpu_data();
  const Dtype* item_feature_img = bottom[0]->cpu_data();
  const Dtype* itact_data_ = bottom[1]->cpu_data();
  const Dtype* itact_count_ = bottom[2]->cpu_data();
  const Dtype* global_bias = this->blobs_[2]->cpu_data();
  // user feature buffer
  Dtype* user_feature_buf = user_feature_buffer_.mutable_cpu_data();
  Dtype* item_feature_buf = item_feature_mixed_.mutable_cpu_data();
  Dtype* itact_pred_ = (*top)[0]->mutable_cpu_data();

  int item_offset = 0, rating_size = 0;
  int item_real_id = 0, userid = 0, rating_idx = 0;
  // relative itemid_version
  for (int itemid = 0; itemid < itact_item_; ++itemid ) {
    item_offset = itact_count_[itemid*2];
    rating_size = itact_count_[itemid*2+1];
    item_real_id = itact_data_[item_offset*2];
    caffe_copy(num_latent_, item_feature + item_real_id*num_latent_, item_feature_buf + itemid*num_latent_);
    caffe_cpu_axpby(num_latent_, img_weight_, item_feature_img + itemid*num_latent_, feature_weight_, item_feature_buf + itemid*num_latent_);

    // LOG(INFO) << "itemid:" << itemid << " offset:" << item_offset << " rating_size: " << rating_size;
    for (int rating_cnt = 0; rating_cnt < rating_size; ++rating_cnt) {
      rating_idx = item_offset + rating_cnt;
      userid = static_cast<int>(itact_data_[rating_idx*2+1]);
      caffe_copy(num_latent_, user_feature + userid*num_latent_, user_feature_buf + rating_cnt*num_latent_);
      // LOG(INFO) << "itemid:" << itemid << " userid:" << userid;
    }

    caffe_cpu_gemv(CblasNoTrans, rating_size, num_latent_,
      Dtype(1.0), user_feature_buf, item_feature_buf + itemid*num_latent_, Dtype(0.),
      itact_pred_ + item_offset);

    // checking user_feature_buf
    // LOG(INFO) << "checking user_feature_buf";
    // for (int i = 0; i < rating_size; i++) {
    //   for (int j = 0; j < num_latent_; j++) {
    //     std::cout << user_feature_buf[i*num_latent_+j] << "\t";
    //   }
    //   std::cout << std::endl;
    // }

    // LOG(INFO) << "Item feature";
    // for (int j = 0; j < num_latent_; j++) {
    //   std::cout << item_feature_buf[itemid*num_latent_+j] << "\t";
    // } 
    // std::cout << std::endl;

    // checking prediction
    // LOG(INFO) << "Rating prediction";
    // for (int j = 0; j < rating_size; j++) {
    //   std::cout << itact_pred_[item_offset+j] << "\t";
    // } 
    // std::cout << std::endl;

  }
  if (bias_term_) {
    caffe_add_scalar(num_rating_, global_bias[0], itact_pred_);
  }
  // set the extra space in itact_pred_ to 0
  if (num_rating_ < max_rating_size_) {
    int extra_length = max_rating_size_ - num_rating_;
    caffe_set(extra_length, Dtype(0.), itact_pred_ + num_rating_);
  }
  // // checking prediction
  // LOG(INFO) << "Rating prediction after adding bias " << global_bias[0];
  // LOG(INFO) << "Global bias " << global_bias[0];
  // for (int j = 0; j < num_rating_; j++) {
  //   std::cout << itact_pred_[j] << "\t";
  // } 
  // std::cout << std::endl;
}

/*  given error for each <itemid, userid, error>
    Update U
      user_feature_diff = sum(rating_error*item_feature) w.r.t this user
      User map to get the itemset for each user.
    Update V
      item_feature_diff = sum(rating_error*user_feature) w.r.t this item
    Update global bias
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // LOG(INFO) << "Backward_cpu";

  // checking loss
  // const Dtype* rating_diff = top[0]->cpu_diff();
  // LOG(INFO) << "loss ";
  // for (int j = 0; j < num_rating_; j++) {
  //   std::cout << rating_diff[j] << "\t";
  // } 
  // std::cout << std::endl;

  Backward_User_cpu(top, propagate_down, bottom);
  Backward_Item_cpu(top, propagate_down, bottom);
  Backward_Item_img_cpu(top, propagate_down, bottom);
  // update global bias. 
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* rating_diff = top[0]->cpu_diff();
    Dtype* bias_diff = this->blobs_[2]->mutable_cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasNoTrans, 1, num_rating_, Dtype(1.), rating_diff,
        bias_multiplier_.cpu_data(), Dtype(0.),
        bias_diff);
  }
}

/*  Traverse each userid,
      find corresponding rating loss, itemid 
      fill item_feature_buf
      gemv and normalize
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_User_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // LOG(INFO) << "Backward_User_cpu";
  // bp diff to user feature in blob_[0]
  if (this->param_propagate_down_[0]) {
    const Dtype* rating_diff = top[0]->cpu_diff();
    Dtype* user_feature_diff = this->blobs_[0]->mutable_cpu_diff(); // Target
    const Dtype* item_feature = item_feature_mixed_.cpu_data(); // feature already combined with img feature
    const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
    Dtype* item_feature_buf = item_feature_buffer_.mutable_cpu_data();

    int userid = 0, itemid = 0, rating_idx = 0, rating_size = 0;
    Dtype loss = 0;
    // traverse each user, #user gemv
    map<int, vector<int> >::iterator iter;
    for (iter = user2itemid.begin(); iter!=user2itemid.end(); ++iter) {
      userid = iter->first;
      vector<int>& ratingset = iter->second;
      rating_size = ratingset.size();
      rating_buffer_.Reshape(1,1,1,rating_size);
      Dtype* rating_buf = rating_buffer_.mutable_cpu_data();

      // std::cout << "userid:" << userid << " rating_size:" << rating_size << std::endl;
      for (int i = 0; i < rating_size; ++i) {
        rating_idx = ratingset[i];
        itemid = itemid2relative_id[static_cast<int>(itact_data_[rating_idx*2])]; // relative itemid
        // fill buffer
        loss = rating_diff[rating_idx];
        rating_buf[i] = loss;
        caffe_copy(num_latent_, item_feature + itemid*num_latent_, item_feature_buf + i*num_latent_);
        // std::cout << "rating_idx:" << rating_idx << " itemid:" << itemid << " loss:" << loss << std::endl;
      }
      caffe_cpu_gemv(CblasTrans, rating_size, num_latent_,
        Dtype(1.0), item_feature_buf, rating_buf, Dtype(0.),
        user_feature_diff + userid*num_latent_);

      // LOG(INFO) << "item_feature_buf";
      // for (int i = 0; i < rating_size; ++i) {
      //   for (int j = 0; j < num_latent_; ++j) {
      //     std::cout << item_feature_buf[i*num_latent_ + j] << "\t";
      //   }
      //   std::cout << std::endl;
      // }

      // LOG(INFO) << "loss";
      // for (int i = 0; i < rating_size; i++) {
      //   std::cout << rating_buf[i] << "\t";
      // }
      // std::cout << std::endl;

      // LOG(INFO) << "user_feature_diff";
      // for (int i = 0; i < num_latent_; i++) {
      //   std::cout << user_feature_diff[userid*num_latent_ + i] << "\t";
      // }
      // std::cout << std::endl;

      // LOG(INFO) << "user_feature";
      // std::cout << "user_feature" << std::endl;
      // const Dtype* user_feature = this->blobs_[0]->cpu_data();
      // for (int i = 0; i < num_latent_; i++) {
      //   std::cout << user_feature[userid*num_latent_ + i] << "\t";
      // }
      // std::cout << std::endl;
    } // ~ for each userid
  }
}

/*  
    For each itemid,
      find associated rating loss, userid
      fill user_feature_buf
      gemv and normalize
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_Item_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // LOG(INFO) << "Backward_Item_cpu";
  // bp diff to item feature in blobs_[1]
  if (this->param_propagate_down_[1]) {
    // LOG(INFO) << "Backward_Item_cpu begin";
    if (feature_weight_ <= 0)
      return;
    if (!gen_item_diff_) {
      // LOG(INFO) << "Calculating";
      const Dtype* rating_diff = top[0]->cpu_diff();
      Dtype* item_feature_diff = this->blobs_[1]->mutable_cpu_diff(); // Target
      const Dtype* user_feature = this->blobs_[0]->cpu_data(); 
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      Dtype* user_feature_buf = user_feature_buffer_.mutable_cpu_data();

      int item_offset = 0, rating_size = 0;
      int item_real_id = 0, userid = 0, rating_idx = 0;
      // relative itemid_version
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        rating_size = itact_count_[itemid*2+1];
        item_real_id = itact_data_[item_offset*2];
        // loss is inherently continuous
        // int item_real_id = static_cast<int>(itact_data_[item_offset*2]);
        // std::cout << "itemid:" << itemid << " item_real_id:" << item_real_id << " offset:" << item_offset << " rating_size: " << rating_size << std::endl;;
        for (int rating_cnt = 0; rating_cnt < rating_size; ++rating_cnt) {
          rating_idx = item_offset + rating_cnt; // simple mapping
          userid = static_cast<int>(itact_data_[rating_idx*2+1]);
          caffe_copy(num_latent_, user_feature + userid*num_latent_, user_feature_buf + rating_cnt*num_latent_);
          // std::cout << "rating_idx:" << rating_idx  << " itemid:" << itemid << " userid:" << userid << std::endl;
        }
        caffe_cpu_gemv(CblasTrans, rating_size, num_latent_,
          feature_weight_, user_feature_buf, rating_diff + item_offset, Dtype(0.),
          item_feature_diff + item_real_id*num_latent_);

        gen_item_diff_ = true; // indicate we have computed item diff

        // LOG(INFO) << "user_feature_buf";
        // for (int i = 0; i < rating_size; ++i) {
        //   for (int j = 0; j < num_latent_; ++j) {
        //     std::cout << user_feature_buf[i*num_latent_ + j] << "\t";
        //   }
        //   std::cout << std::endl;
        // }

        // LOG(INFO) << "loss";
        // for (int i = 0; i < rating_size; i++) {
        //   std::cout << rating_diff[item_offset+i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature_diff";
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature_diff[item_real_id*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature";
        // std::cout << "item_feature" << std::endl;
        // const Dtype* item_feature = this->blobs_[1]->cpu_data(); // Target
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature[item_real_id*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;
      }
    } else {
      // LOG(INFO) << "copying";
      // the diff is already calculated in bottom[0]->cpu_diff()
      const Dtype* item_feature_diff_source = (*bottom)[0]->cpu_diff();
      Dtype* item_feature_diff = this->blobs_[1]->mutable_cpu_diff(); // Target
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      int item_offset = 0, item_real_id = 0;
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        item_real_id = itact_data_[item_offset*2];
        caffe_cpu_scale(num_latent_, feature_weight_/img_weight_, item_feature_diff_source + itemid*num_latent_, item_feature_diff + item_real_id*num_latent_);

        // LOG(INFO) << "item_feature_diff";
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature_diff[item_real_id*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature";
        // std::cout << "item_feature" << std::endl;
        // const Dtype* item_feature = this->blobs_[1]->cpu_data(); // Target
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature[item_real_id*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;
      } // ~ for
    } // ~ if (!gen_item_diff_)
  } // ~ if (this->param_propagate_down_[1])
  // LOG(INFO) << "done";
}

/*  
    For each itemid,
      find associated rating loss, userid
      fill user_feature_buf
      gemv and normalize
*/
template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_Item_img_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // LOG(INFO) << "Backward_Item_img_cpu";
  // bp diff to item feature in bottom[0]
  if (propagate_down[0]) {
    // LOG(INFO) << "Backward_Item_img_cpu begin";
    if (img_weight_ <= 0)
      return;
    if (!gen_item_diff_) {
      // LOG(INFO) << "Calculating";
      const Dtype* rating_diff = top[0]->cpu_diff();
      Dtype* item_feature_diff = (*bottom)[0]->mutable_cpu_diff(); // Target
      const Dtype* user_feature = this->blobs_[0]->cpu_data(); 
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      Dtype* user_feature_buf = user_feature_buffer_.mutable_cpu_data();

      int item_offset = 0, rating_size = 0;
      int item_real_id = 0, userid = 0, rating_idx = 0;
      // relative itemid_version
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        rating_size = itact_count_[itemid*2+1];
        item_real_id = itact_data_[item_offset*2];
        // loss is inherently continuous
        // int item_real_id = static_cast<int>(itact_data_[item_offset*2]);
        // std::cout << "itemid:" << itemid << " item_real_id:" << item_real_id << " offset:" << item_offset << " rating_size: " << rating_size << std::endl;;
        for (int rating_cnt = 0; rating_cnt < rating_size; ++rating_cnt) {
          rating_idx = item_offset + rating_cnt; // simple mapping
          userid = static_cast<int>(itact_data_[rating_idx*2+1]);
          caffe_copy(num_latent_, user_feature + userid*num_latent_, user_feature_buf + rating_cnt*num_latent_);
          // std::cout << "rating_idx:" << rating_idx  << " itemid:" << itemid << " userid:" << userid << std::endl;
        }
        caffe_cpu_gemv(CblasTrans, rating_size, num_latent_,
          img_weight_, user_feature_buf, rating_diff + item_offset, Dtype(0.),
          item_feature_diff + itemid*num_latent_);

        gen_item_diff_ = true; // indicate we have computed item diff

        // LOG(INFO) << "user_feature_buf";
        // for (int i = 0; i < rating_size; ++i) {
        //   for (int j = 0; j < num_latent_; ++j) {
        //     std::cout << user_feature_buf[i*num_latent_ + j] << "\t";
        //   }
        //   std::cout << std::endl;
        // }

        // LOG(INFO) << "loss";
        // for (int i = 0; i < rating_size; i++) {
        //   std::cout << rating_diff[item_offset+i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature_diff";
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature_diff[itemid*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature";
        // std::cout << "item_feature" << std::endl;
        // const Dtype* item_feature = (*bottom)[0]->cpu_data(); // Target
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature[itemid*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;
      }
    } else {
      // LOG(INFO) << "copying";
      // the diff is already calculated in blobs_[1]->cpu_diff()
      const Dtype* item_feature_diff_source = this->blobs_[1]->cpu_diff(); 
      Dtype* item_feature_diff = (*bottom)[0]->mutable_cpu_diff(); // Target
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      int item_offset = 0, item_real_id = 0;
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        item_real_id = itact_data_[item_offset*2];
        caffe_cpu_scale(num_latent_, img_weight_/feature_weight_, item_feature_diff_source + item_real_id*num_latent_, item_feature_diff + itemid*num_latent_);

        // LOG(INFO) << "item_feature_diff";
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature_diff[itemid*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;

        // LOG(INFO) << "item_feature";
        // std::cout << "item_feature" << std::endl;
        // const Dtype* item_feature = (*bottom)[0]->cpu_data(); // Target
        // for (int i = 0; i < num_latent_; i++) {
        //   std::cout << item_feature[itemid*num_latent_ + i] << "\t";
        // }
        // std::cout << std::endl;
      } // ~ for
    } // ~ if (!gen_item_diff_)
  } // ~ if (propagate_down[0])
  // LOG(INFO) << "done";
}

#ifdef CPU_ONLY
STUB_GPU(MatrixFactorizeLayer);
#endif

INSTANTIATE_CLASS(MatrixFactorizeLayer);

}  // namespace caffe
