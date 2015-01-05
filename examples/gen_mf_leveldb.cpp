// Copyright 2014 Kaixiang MO
// This program converts a set of images to a leveldb by storing them as DatumInteraction
// proto buffers.
// Usage:
// 1. rating_file 2. map_itemid_file 3.item_category 4.output_leveldb
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
#include <map>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <typeinfo>

using namespace caffe;
using std::vector;
using std::string;

const int MAX_ITEM_RATING = 500;
const int SIZE=256;

bool str2int (long int &i, string s)
{
    std::stringstream ss(s);
    ss >> i;
    if (ss.fail()) {
        // not an integer
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
	if (argc != 6) {
		printf("Generate leveldb for interaction rating prediction.\n"
				"Usage:\n"
				"    1.rating_file 2.map_itemid_file 3.item_category 4.image-folder 5.output_leveldb");
		return 0;
	}
	
	// read in 4 parameters
	string rating_file(argv[1]);
	string map_item_file(argv[2]); 
	string item_category(argv[3]);
	string image_folder(argv[4]);
	string output_leveldb(argv[5]);

	// itemid_mapper
	map<long int, long int> itemid_mapper;
	map<long int, long int> item_category_mapper;
	// reading itemid_mapper and item_category_mapper into memory
	string line, item_id_str, item_name;
	long int itemid, item_id_long, category_id;
	long int userid, imp, click;

	std::ifstream itemid_infile(map_item_file.c_str());
	while (getline(itemid_infile,line)) {
		std::istringstream lineString(line);
		getline(lineString, item_name,',');
		getline(lineString, item_id_str,',');
		// convert to long int
		str2int(item_id_long, item_name);
		str2int(itemid, item_id_str);
		// std::cout<< item_name << "\t" << item_id_str << std::endl;
		// std::cout<< item_id_long << "\t" << itemid << std::endl;
		itemid_mapper[itemid] = item_id_long;
	}
	std::cout << "itemid_mapper finished." << std::endl;

	std::ifstream category_infile(item_category.c_str());
	while (getline(category_infile,line)) {
		std::istringstream lineString(line);
		getline(lineString, item_name,',');
		getline(lineString, item_id_str,',');
		// convert to long int
		str2int(itemid, item_name);
		str2int(category_id, item_id_str);
		// std::cout<< item_name << "\t" << item_id_str << std::endl;
		// std::cout<< itemid << "\t" << category_id << std::endl;
		item_category_mapper[itemid] = category_id;
	}
	std::cout << "item_category_mapper finished." << std::endl;

	LOG(INFO) << "Using leveldb " << output_leveldb;
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 1024*1024*100; // 100 MB
	options.max_open_files = 100;
	options.block_size = 1024*1024*100; // 100 MB
	leveldb::Status status = leveldb::DB::Open(options, output_leveldb.c_str(), &db);
	CHECK(status.ok());

	DatumInteraction datumItact;
	Datum* datum;
	std::cout << "Generation begin" << std::endl;

	string itemid_str, userid_str, imp_str, click_str, img_name;
	long int itemid_last = 0;
	long int batch_cnt = 0, rating_cnt = 0, datum_cnt = 0;
	std::ifstream rating_infile(rating_file.c_str());
	while (getline(rating_infile,line)) {
		std::istringstream lineString(line);
		getline(lineString, itemid_str,',');
		str2int(itemid, itemid_str);
		getline(lineString, userid_str,',');
		str2int(userid, userid_str);
		getline(lineString, imp_str,',');
		str2int(imp, imp_str);
		getline(lineString, click_str,',');
		str2int(click, click_str);
		item_id_long = itemid_mapper[itemid];
		category_id = item_category_mapper[itemid];

		// std::cout << itemid << "\t" << userid << "\t" << click << "\titem_id_long:" << item_id_long
		// 	<< "\tcategory_id:" << category_id << std::endl;

		// we have itemid, userid, click
		// if we just started / meet a new itemid / current rating number exceeds maximum
		// 		make a new datum, read in the image.
		if (itemid_last != itemid || rating_cnt>=MAX_ITEM_RATING) {
			// If there is any rating
			if (datumItact.rating_size() > 0) {
				stringstream ss;
				ss << datum_cnt;
				db->Put(leveldb::WriteOptions(), ss.str(), datumItact.SerializeAsString());
				datum_cnt ++;

				if (datum_cnt%1000==0) {
					std::cout << "datum number " << datum_cnt << std::endl;
				}
			}

			// clear datumItact
			datum = datumItact.mutable_datum();
			datum->clear_float_data();
			datum->clear_data();
			datumItact.clear_itemid();
			datumItact.clear_userid();
			datumItact.clear_rating();

			datum = datumItact.mutable_datum();
			stringstream ss;
			ss << image_folder + "/" << item_id_long << ".jpg";
			img_name = ss.str();
			// img_name = image_folder + "/" + string(itemid) + ".jpg";
			// resize all image to SIZE*SIZE
			if (!ReadImageToDatumWithSameAspectRatio(img_name, category_id, SIZE, SIZE, true, datum)) {
				std::cout << img_name << " is broken" << std::endl;
				continue;
			}

			if (itemid_last == itemid) {
				batch_cnt ++;
			} else {
				itemid_last = itemid;
				batch_cnt = 0;
			}
			rating_cnt = 0;
		}
		datumItact.add_itemid(itemid);
		datumItact.add_userid(userid);
		if (click>0) {
			datumItact.add_rating(1);
		} else {
			datumItact.add_rating(0);
		}
		rating_cnt ++;
	}

	// adding remainint 
	if (datumItact.rating_size() > 0) {
		stringstream ss;
		ss << datum_cnt;
		db->Put(leveldb::WriteOptions(), ss.str(), datumItact.SerializeAsString());
		datum_cnt ++;
		if (datum_cnt%1000==0) {
			std::cout << "datum number " << datum_cnt << std::endl;
		}
	}

	delete db;
	std::cout << "Finished. Total " << datum_cnt << " datums." << std::endl;

	// // reading the leveldb
	// shared_ptr<leveldb::DB> db_;
	// shared_ptr<leveldb::Iterator> iter_;

 //  leveldb::DB* db_temp;
 //  leveldb::Options options1 = GetLevelDBOptions();
 //  options1.create_if_missing = false;
 //  LOG(INFO) << "Opening leveldb " << argv[1];
 //  leveldb::Status status1 = leveldb::DB::Open(
 //      options1, output_leveldb, &db_temp);
 //  CHECK(status1.ok()) << "Failed to open leveldb "
 //                     << output_leveldb << std::endl
 //                     << status1.ToString();

 //  db_.reset(db_temp);
 //  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
 //  iter_->SeekToFirst();

 //  DatumInteraction datumItract;
 //  while(iter_->Valid()) {
 //  	datumItract.ParseFromString(iter_->value().ToString());
 //  	for (int i = 0; i < datumItract.rating_size(); i++) {
 //  		std::cout << datumItract.itemid(i) << " " << datumItract.userid(i) << " " << datumItract.rating(i) << std::endl;
 //  	}
 //  	const string& data = datumItract.datum().data();
 //  	for (int j = 1; j < 10; j++) {
 //  		std::cout << static_cast<int>(static_cast<uint8_t>(data[j])) << "\t";
 //  	}
 //  	std::cout << std::endl;
 //  	iter_->Next();
 //  }

	return 0;
}