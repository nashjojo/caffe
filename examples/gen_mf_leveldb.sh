#!/bin/sh
folder="/home/nash/Documents/TmallImgCtr/data/ImageMF/mapping"
target_file="/home/nash/Documents/TmallImgCtr/data/ImageMF/worksplace/dataset/full_itact_data"

# if [ -d $target_file ]; then
# 	echo $target_file removed 
# 	rm $target_file -r
# fi

# echo testing
# ./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt.head ${folder}/auction_id.txt.head ${folder}/item_mapped.txt.head /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ./examples/temp_itact_data

# echo ${target_file}
# ./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images $target_file
# echo ${target_file}
# ./build/examples/gen_mf_leveldb.bin ${folder}/train_0.5.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ${target_file}_0.5
# echo ${target_file}_test
# ./build/examples/gen_mf_leveldb.bin ${folder}/test_mapped.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ${target_file}_test


echo testing img and iteraction data
./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt.tail ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ./examples/temp_itact_data

# netflix dataset.
# echo movielens 
# ./build/examples/gen_mf_leveldb.bin /home/nash/Documents/code_BPMF/train_vec.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ./examples/movielens_itact_data
# ./build/examples/gen_mf_leveldb.bin /home/nash/Documents/code_BPMF/probe_vec.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ./examples/movielens_itact_data_test


# toy dataset
# toy_folder="/home/nash/Documents/TmallImgCtr/data/ImageMF/toy_exp"
# echo Toy dataset
# ./build/examples/gen_mf_leveldb.bin ${toy_folder}/train_mapped.txt.head ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ${toy_folder}/toy_dataset