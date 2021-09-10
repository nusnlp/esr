#!/bin/bash

. $(dirname $BASH_SOURCE)/../../../../../script/esr/base.sh

device=0,1
transformer=roberta-large
dataset=dataset_semcor
seed=42

prefix='a100_'

train_args='
--per_device_train_batch_size 16
--per_device_eval_batch_size 16
--learning_rate 8.5e-6
--input_limit 348
'
test_args='
--per_device_eval_batch_size 16
'

$script/train.sh $device $transformer $dataset $seed "$prefix" "$train_args"
$script/test.sh $device $transformer $dataset $seed "$prefix" "$train_args" "$test_args"
