# Financial Text Summarization using RL policies
This repository contains the code for our Financial Texts Summarization task, realized for Deep Natural Language Process class(2021-2022).
Our task was to re-produce the work made by [Zmander](https://arxiv.org/abs/1805.11080) who managed to rank 2nd in [FNS2021 competition](http://wp.lancs.ac.uk/cfie/fns2021/). In this work, we will exploit long texts' summarization task combining an extraction and abstraction approaches by a Reinforcement Learning policy. In addition to that, we propose a distributional analysis to understand which are the most salient parts of the documents and to cut them according to the distribution we found. Furthemore, we extend the task to CNN's headline generation and we use the model to FNS2022's dataset which is composed by three different languages: English, Spanish and Greek. 
The pipeline we propose, and you can reproduce, is the following:
1. Data preprocessing, comprising documents'cut, corpus generation, etc.
2. Extractor, Abstractor and RL models training
3. Model evaluation



## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

## Datasets Download
You can download the datasets we used from the following links
1. [FNS 2022 Dataset](https://drive.google.com/drive/folders/1lvvMgDBR1WfxrHJmDCo58XprLBNE3U5L?usp=sharing)
2. [CNN Headlines Generation Dataset](https://drive.google.com/file/d/1ReQOXmjatCKuBfSGvdv24Xmju_Qq4ral/view?usp=sharing)

## Execution guide
In the following, we will give you a suggestion on how to re-run our work. In this way you can check our results and change the training settings if you like. If you like, a colab notebook with the most salient steps is provided at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ow_iprH3OKqKGx_sK4l-nc7h_GH72F0L).
## Distribution
This step can be avoided. In fact, if it is not run, the cut is performed using to first 1000 sentences. If you would like to have a look at the distribution of importance in your documents you should run:
```
%run preprocess/distribution_analysis.py --data <DATASET> --language <language> --stage <stage you want to start from> --top_M <top sentences to compute the rouge with> --jit <True if you want to use jit, False otherwise>
```
Note that if the parameter --jit is not specified, the code is run using it.
## Preprocessing pipeline
The following sketch shows an idea of how our preprocessing pipeline works. Note that the documents are cut according to their distribution

<p align="center">
<img src="/images/preprocess.jpg" title="Preprocessing pipeline" width="500" height="200">

## Preprocessing
First of all, you will need to pre-process your data. To do that, you can use the script pipeline.py which is inside the folder called "preprocess". In this way, you will transform you data in ordert to be feasible for the models.
```
!python preprocess/pipeline.py --data <DATASET> --language <language> --max_len <maximum length> --stage <stage_you_want_to_start_from> --jit <True if you want to use jit, False otherwise> --use_distribution <stores true if you want to cut documents according to distribution>
```
Note: if the parameter --jit is not specified, the code is run using it.


## Train extractor
Next step is to train extractor. The image displays the main idea of its architecture.

<p align="center">
<img src="/images/extractor.jpg" alt="Alt text" title="Preprocessing pipeline" width="600" height="300">



To train it run the following cell:
```
!python train_extractor_ml.py --data <DATASET> --language <language> --lstm_layer <number of lstms> --lstm_hidden <number of lstm hidden layers> --batch <batch size> --ckpt_freq <checkpoint frequency> --max_word <words in a sentence are cut according to this parameter> --max_sent <sentences in a document are cut according to this parameter>
```
## Train abstractor
If you want to train abstractor, following line needs to be executed:
```
!python train_abstractor.py --data <DATASET> --language <language> --n_layer <number of layers> --n_hidden <number of hidden layers> --batch <batch size> --ckpt_freq <checkpoint_frequnce>
```
## Train RL model

Last, but not the least, model to be trained is the Reinforcement Learning's agent. The scheme shows the main steps of its functioning idea.

<p align="center">
<img src="/images/reinforcement.jpg" alt="Alt text" title="Preprocessing pipeline" width="500" height="300">

Concerning RL, the parameter --abs_dir passes the abstraction directory to RL model. If you like performing our ablation study, do not pass the parameter --abs dir. If you would like to train RL part, run the following cell:
```
!python train_full_rl.py --data <DATASET> --language <language> --batch <batch size> --abs_dir <directory of the abstractor (use "model\abs"). If you want to perform ablation study, do not pass this argument>
```
## Evaluate the model

In the end, evaluate the model using the script decode_full_model.py:
```
!python decode_full_model.py --data <DATASET> --language <language> --batch <batch size>
```

### Results
You should get the following results if you cut the documents according to their distribution.

| Language | HypPar                              | R-type    | R-1         | R-2         | R-L         |
|----------|-------------------------------------|-----------|-------------|-------------|-------------|
| English  | n_hidden=128 n_LSTM=1 batch_size=4  | F1 | 0.332 | 0.118 | 0.326 |
|   |   | Recall | 0.315 | 0.117 | 0.309 |
| Greek    | n_hidden=256 n_LSTM=2 batch_size=4 | F1 | 0.489 | 0.311  | 0.479 |
|   |   | Recall | 0.227 | 0.140 | 0.224 |
| Spanish  | n_hidden=128 n_LSTM=1 batch_size=4  | F1 | 0.340 | 0.094 | 0.334 |
|   |   | Recall | 0.292 | 0.081 | 0.286 |


