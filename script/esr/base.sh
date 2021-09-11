#!/bin/bash

. $(dirname $BASH_SOURCE)/../config.sh

export LANG="C.UTF-8"

wef_train=$data/WSD_Evaluation_Framework/Training_Corpora
wef_test=$data/WSD_Evaluation_Framework/Evaluation_Datasets
ufsac=$data/ufsac-public-2.1
fews=$(dirname $BASH_SOURCE)/../../experiment/fews/xml

script=$(dirname $BASH_SOURCE)/../esr
code=$(dirname $BASH_SOURCE)/../../code/esr
experiment=$(dirname $BASH_SOURCE)/../../experiment/esr

. $conda_path/etc/profile.d/conda.sh
conda activate $virtual_env

function device_count {
    devices=($(echo $1 | tr ',' '\n'))
    echo ${#devices[@]}
}

function check_distributed {
    n_device=$(device_count $1)
    if [ $n_device -gt 1 ]; then
        echo "-m torch.distributed.launch --nproc_per_node=$n_device"
    else
        echo ""
    fi
}
