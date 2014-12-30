// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as DatumPosNeg
// proto buffers.
// Usage:
// 		0 						1 					2 				3 			4 				
//    convert_multi ROOTFOLDER/ LISTFILE 	DB_NAME shuffle 
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   urls			adid        <label, weight>
//   201.jpg	20013123    
//   ....
// We will shuffle the file if the last digit is 1. 
#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "boost/scoped_ptr.hpp"

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <typeinfo>

using namespace caffe;
using std::vector;
using std::string;

const int NUM_LABEL=3;
const int MIN_IMG_SIZE=160;
const int MAX_IMG_SIZE=192;
const int WRITE_BATCH_SIZE = 20000;

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Make a temp leveldb for interaction_data_layer_test.\n"
				"Usage:\n"
				"    interaction_data_layer_test dbname");
		return 0;
	}
	
	LOG(INFO) << "Using temporary leveldb " << argv[1];
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;

	options.write_buffer_size = 1024*1024*100; // 100 MB
	options.max_open_files = 100;
	options.block_size = 1024*1024*100; // 100 MB

	leveldb::Status status = leveldb::DB::Open(options, argv[1], &db);
	CHECK(status.ok());

	DatumInteraction datumItact;
	Datum* datum;
	std::cout << "begin looping" << std::endl;
	for (int i = 0; i < 5; ++i) {
		// DatumInteraction datumItact;
		// std::cout << i << " DatumInteraction datumItact;" << std::endl;
		// Datum datum;
		// std::cout << i << " Datum datum;" << std::endl;
		// datumItact.set_allocated_datum(&datum);
		
		// setting userid and ratings
		for (int j = 0; j < (6-i); ++j) {
			datumItact.add_itemid(10-i);
			datumItact.add_userid(j);
			datumItact.add_rating(i*j);
		}
		std::cout << i << " rating_size " << datumItact.rating_size() << std::endl;

		datum = datumItact.mutable_datum();
		datum->set_label(i%2);
		datum->set_channels(2);
		datum->set_height(3);
		datum->set_width(4);
		std::cout << i << " datum->set_width(4)" << std::endl;
		std::string* data = datum->mutable_data();
		for (int j = 0; j < 24; ++j) {
			int temp_data = i;
			data->push_back(static_cast<uint8_t>(temp_data));
		}
		std::cout << i << " data_size " << datum->data().length() << std::endl;

		stringstream ss;
		ss << i;
		db->Put(leveldb::WriteOptions(), ss.str(), datumItact.SerializeAsString());
		datumItact.Clear();
		std::cout << i << " end of loop" << std::endl;
	}
	delete db;


	// reading the leveldb
	shared_ptr<leveldb::DB> db_;
	shared_ptr<leveldb::Iterator> iter_;

  leveldb::DB* db_temp;
  leveldb::Options options1 = GetLevelDBOptions();
  options1.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status1 = leveldb::DB::Open(
      options1, argv[1], &db_temp);
  CHECK(status1.ok()) << "Failed to open leveldb "
                     << argv[1] << std::endl
                     << status1.ToString();

  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  DatumInteraction datumItract;
  while(iter_->Valid()) {
  	datumItract.ParseFromString(iter_->value().ToString());
  	std::cout << datumItract.itemid(0) << " " << datumItract.userid(0) << " " << datumItract.datum().label() << std::endl;
  	const string& data = datumItract.datum().data();
  	for (int j = 1; j < 24; j++) {
  		std::cout << static_cast<int>(static_cast<uint8_t>(data[j])) << "\t";
  	}
  	std::cout << std::endl;
  	iter_->Next();
  }

	return 0;
}
