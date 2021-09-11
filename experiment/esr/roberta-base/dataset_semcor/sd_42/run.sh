#!/bin/bash

. $(dirname $BASH_SOURCE)/../../../../../script/esr/base.sh

device=0
transformer=roberta-base
dataset=dataset_semcor
seed=42

prefix='rtx3090_'

train_args='
--per_device_train_batch_size 32
--per_device_eval_batch_size 32
--learning_rate 8.5e-6
'
test_args='
--per_device_eval_batch_size 32
'

$script/train.sh $device $transformer $dataset $seed "$prefix" "$train_args"
$script/test.sh $device $transformer $dataset $seed "$prefix" "$train_args" "$test_args"
