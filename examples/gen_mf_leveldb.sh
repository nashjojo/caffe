#!/bin/sh
folder="/home/nash/Documents/TmallImgCtr/data/ImageMF/mapping"
target_file="/home/nash/Documents/TmallImgCtr/data/ImageMF/worksplace/imageMF/full_itact_data"

# if [ -d $target_file ]; then
# 	echo $target_file removed 
# 	rm $target_file -r
# fi

# ./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt.head ${folder}/auction_id.txt.head ${folder}/item_mapped.txt.head /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images $target_file

echo ${target_file}
./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images $target_file
echo ${target_file}_test
./build/examples/gen_mf_leveldb.bin ${folder}/train_mapped.txt ${folder}/auction_id.txt ${folder}/item_mapped.txt /home/nash/Documents/TmallImgCtr/data/ImageMF/original/images ${target_file}_test
