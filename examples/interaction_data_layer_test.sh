#!/bin/sh
target_file=./examples/temp_itact_data
if [ -d $target_file ]; then
	echo $target_file removed 
	rm $target_file -r
fi
./build/examples/interaction_data_layer_test.bin $target_file
