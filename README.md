# Improved Word Sense Disambiguation with Enhanced Sense Representations

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-word-sense-disambiguation-with/word-sense-disambiguation-on-supervised)](https://paperswithcode.com/sota/word-sense-disambiguation-on-supervised?p=improved-word-sense-disambiguation-with)

This repository contains codes and scripts to build enhanced sense representations for word sense disambiguation.

If you use this code for your work, please cite this [paper](https://aclanthology.org/2021.findings-emnlp.365.pdf):
```
@inproceedings{song-etal-2021-improved-word,
    title = "Improved Word Sense Disambiguation with Enhanced Sense Representations",
    author = "Song, Yang  and
      Ong, Xin Cai  and
      Ng, Hwee Tou  and
      Lin, Qian",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    url = "https://aclanthology.org/2021.findings-emnlp.365",
    pages = "4311--4320"
}
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
 * [UFSAC](https://drive.google.com/file/d/1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO)
 * [FEWS](https://nlp.cs.washington.edu/fews/)

Setting up variables
------------

You need to modify `script/config.sh` according to your environment.
Set `data` variable to the top directory where all the datasets are stored.

Processing FEWS
------------

```
bash experiment/fews/run.sh
```

Using trained models
------------

You can train the models from scratch.
Alternatively, you can use our [trained models](https://drive.google.com/file/d/1c8yooOoXsnIgJi0-To7xKNmYU-CugaeL/view?usp=sharing).

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

For ESR on SemCor and WNGC with `roberta-base`:
```
bash experiment/esr/roberta-base/dataset_semcor_wngc/sd_42/run.sh
```

For ESR on SemCor and WNGC with `roberta-large`:
```
bash experiment/esr/roberta-large/dataset_semcor_wngc/sd_42/run.sh
```

For ESR on FEWS with `roberta-base`:
```
bash experiment/esr/roberta-base/dataset_fews/sd_42/run.sh
```

For ESR on FEWS with `roberta-large`:
```
bash experiment/esr/roberta-large/dataset_fews/sd_42/run.sh
```
