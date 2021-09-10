# Improved Word Sense Disambiguation with Enhanced Sense Representations
This repository contains codes and scripts to build enhanced sense representations for word sense disambiguation.

If you use this code for your work, please cite this [paper]():
```
```

Requirements
------------

* python==3.8.8
* pytorch==1.9.0
* transformers==4.6.1
* nltk==3.6.2

Downloading Datasets
------------

You need to download the following datasets:

 * [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval)

Setting up variables
------------

You need to modify `script/config.sh` according to your environment.
Set `data` variable to the top directory where all the datasets are stored.

Running Experiments
------------

For ESR on SemCor with `roberta-base`:
```
bash experiment/esr/roberta-base/dataset_semcor/sd_42/run.sh
```

For ESR on SemCor with `roberta-large`:
```
bash experiment/esr/roberta-large/dataset_semcor/sd_42/run.sh
```
