#!/bin/bash

. $(dirname $BASH_SOURCE)/base.sh

device=$1
transformer=$2
dataset=$3
seed=$4
prefix=$5
train_args=$6
test_args=$7

cur=$experiment/$transformer/$dataset/sd_$seed
train_output=$cur/$prefix$(python $code/abbreviate_args.py $train_args)
test_output=$train_output/$(python $code/abbreviate_args.py $test_args)

pred=$test_output/predictions

log_file=$test_output/result.txt
mkdir -p $test_output
> $log_file

for test in semeval2007 senseval2 senseval3 semeval2013 semeval2015; do

    echo $test | tee -a $log_file

    pred_path=$pred/$test
    pred_file=$pred_path/predictions_softmax.txt

    CUDA_VISIBLE_DEVICES=$device python $code/main.py \
        --seed $seed \
        --cache_dir $cache \
        --model_name $transformer \
        --model_path $train_output/model \
        --output_dir $train_output/model \
        --dataset_index $dataset \
        --do_predict \
        --wsd_xml $wef_test/$test/$test.data.xml \
        --pred_file $pred_file \
        $test_args

    python $code/softmax_to_key.py \
        --softmax_file $pred_file \
        --key_file $pred_path/predictions_key.txt

    $java_path/java -cp $wef_test:. Scorer $wef_test/$test/$test.gold.key.txt $pred_path/predictions_key.txt | tee -a $log_file
done
