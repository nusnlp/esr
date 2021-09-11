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

head -n -1 $xml/dev/dev.few-shot.data.xml > $xml/dev/dev.data.xml
tail -n +3 $xml/dev/dev.zero-shot.data.xml >> $xml/dev/dev.data.xml
cat $xml/dev/dev.few-shot.gold.key.txt > $xml/dev/dev.gold.key.txt
cat $xml/dev/dev.zero-shot.gold.key.txt >> $xml/dev/dev.gold.key.txt

head -n -1 $xml/test/test.few-shot.data.xml > $xml/test/test.data.xml
tail -n +3 $xml/test/test.zero-shot.data.xml >> $xml/test/test.data.xml
cat $xml/test/test.few-shot.gold.key.txt > $xml/test/test.gold.key.txt
cat $xml/test/test.zero-shot.gold.key.txt >> $xml/test/test.gold.key.txt
