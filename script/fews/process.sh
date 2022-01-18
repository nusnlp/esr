#!/bin/bash

. $(dirname $BASH_SOURCE)/../config.sh

export LANG="C.UTF-8"

xml=$1

fews=$data/fews

code=$(dirname $BASH_SOURCE)/../../code/fews

. $conda_path/etc/profile.d/conda.sh
conda activate $virtual_env

python $code/process.py \
    --do_fews_senses \
    --fews_senses $fews/senses.txt

for data in train/train dev/dev.few-shot test/test.few-shot; do

    echo Processing $data.txt ...

    python $code/process.py \
        --do_fews_data \
        --fews_txt $fews/$data.txt \
        --document_id 0 \
        --fews_xml $xml/$data.data.xml \
        --fews_label $xml/$data.gold.key.txt
done

for data in dev/dev.zero-shot test/test.zero-shot; do

    echo Processing $data.txt ...

    python $code/process.py \
        --do_fews_data \
        --fews_txt $fews/$data.txt \
        --document_id 1 \
        --fews_xml $xml/$data.data.xml \
        --fews_label $xml/$data.gold.key.txt
done

for data in dev test; do

    mkdir -p $xml/$data.few-shot
    mv $xml/$data/$data.few-shot.data.xml $xml/$data.few-shot/$data.few-shot.data.xml
    mv $xml/$data/$data.few-shot.gold.key.txt $xml/$data.few-shot/$data.few-shot.gold.key.txt

    mkdir -p $xml/$data.zero-shot
    mv $xml/$data/$data.zero-shot.data.xml $xml/$data.zero-shot/$data.zero-shot.data.xml
    mv $xml/$data/$data.zero-shot.gold.key.txt $xml/$data.zero-shot/$data.zero-shot.gold.key.txt

    head -n -1 $xml/$data.few-shot/$data.few-shot.data.xml > $xml/$data/$data.data.xml
    tail -n +3 $xml/$data.zero-shot/$data.zero-shot.data.xml >> $xml/$data/$data.data.xml
    cat $xml/$data.few-shot/$data.few-shot.gold.key.txt > $xml/$data/$data.gold.key.txt
    cat $xml/$data.zero-shot/$data.zero-shot.gold.key.txt >> $xml/$data/$data.gold.key.txt
done
