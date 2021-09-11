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

mkdir -p $pred/dev
> $pred/dev/predictions_key.txt

for dev in dev.few-shot dev.zero-shot; do

    echo $dev | tee -a $log_file

    pred_path=$pred/$dev
    pred_file=$pred_path/predictions_softmax.txt

    CUDA_VISIBLE_DEVICES=$device python $code/main.py \
        --seed $seed \
        --cache_dir $cache \
        --model_name $transformer \
        --model_path $train_output/model \
        --output_dir $train_output/model \
        --dataset_index $dataset \
        --do_predict \
        --wsd_xml $fews/dev/$dev.data.xml \
        --pred_file $pred_file \
        $test_args

    python $code/softmax_to_key.py \
        --softmax_file $pred_file \
        --key_file $pred_path/predictions_key.txt

    cat $pred_path/predictions_key.txt >> $pred/dev/predictions_key.txt

    $java_path/java -cp $wef_test:. Scorer $fews/dev/$dev.gold.key.txt $pred_path/predictions_key.txt | tee -a $log_file
done

echo dev.full | tee -a $log_file
$java_path/java -cp $wef_test:. Scorer $fews/dev/dev.gold.key.txt $pred/dev/predictions_key.txt | tee -a $log_file

mkdir -p $pred/test
> $pred/test/predictions_key.txt

for test in test.few-shot test.zero-shot; do

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
        --wsd_xml $fews/test/$test.data.xml \
        --pred_file $pred_file \
        $test_args

    python $code/softmax_to_key.py \
        --softmax_file $pred_file \
        --key_file $pred_path/predictions_key.txt

    cat $pred_path/predictions_key.txt >> $pred/test/predictions_key.txt

    $java_path/java -cp $wef_test:. Scorer $fews/test/$test.gold.key.txt $pred_path/predictions_key.txt | tee -a $log_file
done

echo test.full | tee -a $log_file
$java_path/java -cp $wef_test:. Scorer $fews/test/test.gold.key.txt $pred/test/predictions_key.txt | tee -a $log_file
