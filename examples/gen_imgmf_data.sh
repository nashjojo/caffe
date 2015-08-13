#!/bin/sh
stratified="/media/nash/HDD-2T/Ubuntu/TmallImgCtr/data/ImageMF/stratified/TrainTestItemID"
folder="/home/nash/Documents/TmallImgCtr/data/ImageMF/mapping"
fast_img="/media/nash/SSD/Ubuntu/ImageMF/original/images"
target_file="/home/nash/Documents/TmallImgCtr/data/ImageMF/workplace2/dataset/labelid"

# if [ -d $target_file ]; then
# 	echo $target_file removed 
# 	rm $target_file -r
# fi

# echo testing img 
# ../build/examples/gen_imgmf_data.bin ${folder}/train_mapped.txt.tail ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ${target_file}

echo train
../build/examples/gen_imgmf_data.bin ${stratified}/TrainID.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt ${fast_img} ${target_file}/train

echo valid
# ../build/examples/gen_imgmf_data.bin ${stratified}/ValidID.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt ${fast_img} ${target_file}/valid

echo test_head
# ../build/examples/gen_imgmf_data.bin ${stratified}/TestHeadID.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt ${fast_img} ${target_file}/test_head

echo test_tail
../build/examples/gen_imgmf_data.bin ${stratified}/TestTailID.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt ${fast_img} ${target_file}/test_tail

echo test_new
../build/examples/gen_imgmf_data.bin ${stratified}/TestNewID.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt ${fast_img} ${target_file}/test_new