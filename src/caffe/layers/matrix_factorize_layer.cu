#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  if (force_cpu_) {
    return Forward_cpu(bottom, top);
  }

  // LOG(INFO) << "Forward_gpu";
  const Dtype* user_feature = this->blobs_[0]->gpu_data();
  const Dtype* item_feature = this->blobs_[1]->gpu_data();
  const Dtype* item_feature_img = bottom[0]->gpu_data();
  const Dtype* itact_data_ = bottom[1]->cpu_data();				// must use cpu
  const Dtype* itact_count_ = bottom[2]->cpu_data();			// must use cpu
  const Dtype* global_bias = this->blobs_[2]->cpu_data(); // must use cpu
  // user feature buffer
  Dtype* user_feature_buf = user_feature_buffer_.mutable_gpu_data();
  Dtype* item_feature_buf = item_feature_mixed_.mutable_gpu_data();
  Dtype* itact_pred_ = (*top)[0]->mutable_gpu_data();

  int item_offset = 0, rating_size = 0;
  int item_real_id = 0, userid = 0, rating_idx = 0;
  // relative itemid_version
  for (int itemid = 0; itemid < itact_item_; ++itemid ) {
    item_offset = itact_count_[itemid*2];
    rating_size = itact_count_[itemid*2+1];
    item_real_id = itact_data_[item_offset*2];
    caffe_copy(num_latent_, item_feature + item_real_id*num_latent_, item_feature_buf + itemid*num_latent_);
    caffe_gpu_axpby(num_latent_, Dtype(1.0), item_feature_img + itemid*num_latent_, Dtype(1.0), item_feature_buf + itemid*num_latent_);

    // LOG(INFO) << "itemid:" << itemid << " offset:" << item_offset << " rating_size: " << rating_size;
    for (int rating_cnt = 0; rating_cnt < rating_size; ++rating_cnt) {
      rating_idx = item_offset + rating_cnt;
      userid = static_cast<int>(itact_data_[rating_idx*2+1]);
      caffe_copy(num_latent_, user_feature + userid*num_latent_, user_feature_buf + rating_cnt*num_latent_);
      // LOG(INFO) << "itemid:" << itemid << " userid:" << userid;
    }
    caffe_gpu_gemv(CblasNoTrans, rating_size, num_latent_,
      Dtype(1.0), user_feature_buf, item_feature_buf + itemid*num_latent_, Dtype(0.),
      itact_pred_ + item_offset);
  }
  if (bias_term_) {
    caffe_gpu_add_scalar(num_rating_, global_bias[0], itact_pred_);
  }
  // set the extra space in itact_pred_ to 0
  if (num_rating_ < max_rating_size_) {
    int extra_length = max_rating_size_ - num_rating_;
    caffe_gpu_set(extra_length, Dtype(0.), itact_pred_ + num_rating_);
  }
  // checking prediction
  // const Dtype* itact_pred_cpu_ = (*top)[0]->cpu_data();
  // LOG(INFO) << "Global bias " << global_bias[0];
  // for (int j = 0; j < num_rating_; j++) {
  //   if (itact_pred_cpu_[j] > -2) {
  //     std::cout << j << ":" << itact_pred_cpu_[j] << "\t";
  //   }
  // } 
  // std::cout << std::endl;
}

template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (force_cpu_) {
    return Backward_cpu(top, propagate_down, bottom);
  }
  
  // LOG(INFO) << "Backward_gpu";

  // checking loss
  // const Dtype* rating_diff = top[0]->cpu_diff();
  // LOG(INFO) << "loss ";
  // for (int j = 0; j < num_rating_; j++) {
  //   if (abs(rating_diff[j]) > 5e-7) {
  //     std::cout << j << ":" << rating_diff[j] << "\t";
  //   }
  // } 
  // std::cout << std::endl;

  Backward_User_gpu(top, propagate_down, bottom);
  Backward_Item_gpu(top, propagate_down, bottom);
  Backward_Item_img_gpu(top, propagate_down, bottom);
  // update global bias. 
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* rating_diff = top[0]->gpu_diff();
    Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasNoTrans, 1, num_rating_, Dtype(1.), rating_diff,
        bias_multiplier_.gpu_data(), Dtype(0.),
        bias_diff);
  }
}

template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_User_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	// return Backward_User_cpu(top, propagate_down, bottom);
  // LOG(INFO) << "Backward_User_gpu";
  // bp diff to user feature in blob_[0]
  if (this->param_propagate_down_[0]) {
    const Dtype* rating_diff = top[0]->cpu_diff();  // must be cpu
    Dtype* user_feature_diff = this->blobs_[0]->mutable_gpu_diff(); // Target
    const Dtype* item_feature = item_feature_mixed_.gpu_data();
    const Dtype* itact_data_ = (*bottom)[1]->cpu_data();  // must be cpu
    Dtype* item_feature_buf = item_feature_buffer_.mutable_gpu_data();

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
      caffe_gpu_gemv(CblasTrans, rating_size, num_latent_,
        Dtype(1.0) / rating_size, item_feature_buf, rating_buffer_.gpu_data(), Dtype(0.),
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
    }
  }
}

template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_Item_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	// return Backward_Item_cpu(top, propagate_down, bottom);
  // LOG(INFO) << "Backward_Item_gpu";
  // bp diff to item feature in blobs_[1]
  if (this->param_propagate_down_[1]) {
    // LOG(INFO) << "Backward_Item_gpu begin";
    if (!gen_item_diff_) {
      // LOG(INFO) << "Calculating";
      const Dtype* rating_diff = top[0]->gpu_diff();
      Dtype* item_feature_diff = this->blobs_[1]->mutable_gpu_diff(); // Target
      const Dtype* user_feature = this->blobs_[0]->gpu_data(); 
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();  // must be cpu
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data(); // must be cpu
      Dtype* user_feature_buf = user_feature_buffer_.mutable_gpu_data();

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
        caffe_gpu_gemv(CblasTrans, rating_size, num_latent_,
          (Dtype(1.0) / 2) / rating_size, user_feature_buf, rating_diff + item_offset, Dtype(0.),
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
      }
    } else {
      // LOG(INFO) << "copying";
      // the diff is already calculated in bottom[0]->gpu_diff()
      const Dtype* item_feature_diff_source = (*bottom)[0]->gpu_diff();
      Dtype* item_feature_diff = this->blobs_[1]->mutable_gpu_diff(); // Target
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      int item_offset = 0, item_real_id = 0;
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        item_real_id = itact_data_[item_offset*2];
        caffe_copy(num_latent_, item_feature_diff_source + itemid*num_latent_, item_feature_diff + item_real_id*num_latent_);
      } // ~ for
    } // ~ if (!gen_item_diff_)
  } // ~ if (this->param_propagate_down_[1])
  // LOG(INFO) << "done";
}

template <typename Dtype>
void MatrixFactorizeLayer<Dtype>::Backward_Item_img_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // return Backward_Item_cpu(top, propagate_down, bottom);
  // LOG(INFO) << "Backward_Item_img_gpu";
  // bp diff to item feature in bottom[0]
  if (propagate_down[0]) {
    // LOG(INFO) << "Backward_Item_img_gpu begin";
    if (!gen_item_diff_) {
      // LOG(INFO) << "Calculating";
      const Dtype* rating_diff = top[0]->gpu_diff();
      Dtype* item_feature_diff = (*bottom)[0]->mutable_gpu_diff(); // Target
      const Dtype* user_feature = this->blobs_[0]->gpu_data(); 
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();  // must be cpu
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data(); // must be cpu
      Dtype* user_feature_buf = user_feature_buffer_.mutable_gpu_data();

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
        caffe_gpu_gemv(CblasTrans, rating_size, num_latent_,
          (Dtype(1.0) / 2) / rating_size, user_feature_buf, rating_diff + item_offset, Dtype(0.),
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
      }
    } else {
      // LOG(INFO) << "copying";
      // the diff is already calculated in blobs_[1]->gpu_diff()
      const Dtype* item_feature_diff_source = this->blobs_[1]->gpu_diff(); 
      Dtype* item_feature_diff = (*bottom)[0]->mutable_gpu_diff(); // Target
      const Dtype* itact_data_ = (*bottom)[1]->cpu_data();
      const Dtype* itact_count_ = (*bottom)[2]->cpu_data();
      int item_offset = 0, item_real_id = 0;
      for (int itemid = 0; itemid < itact_item_; ++itemid ) {
        item_offset = itact_count_[itemid*2];
        item_real_id = itact_data_[item_offset*2];
        caffe_copy(num_latent_, item_feature_diff_source + item_real_id*num_latent_, item_feature_diff + itemid*num_latent_);
      } // ~ for
    } // ~ if (!gen_item_diff_)
  } // ~ if (propagate_down[0])
  // LOG(INFO) << "done";
}

INSTANTIATE_CLASS(MatrixFactorizeLayer);

}  // namespace caffe
