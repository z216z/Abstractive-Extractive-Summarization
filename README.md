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
In the following, we will give you a suggestion on how to re-run our work. In this way you can check our results and change the training settings if you like.
## Distribution
This step can be avoided. In fact, if it is not run, the cut is performed using to first 1000 sentences. If you would like to have a look at the distribution of importance in your documents you should run:
```
%run preprocess/distribution_analysis.py --data <DATASET> --language <language> --stage <stage you want to start from> --top_M <top sentences to compute the rouge with>
--jit <True if you want to use jit>
```
Note that if the parameter --jit is not specified, the code is run using it.
<img src="/images/preprocessing.png" alt="Alt text" title="Preprocessing pipeline">
## Preprocessing
First of all, you will need to pre-process your data. To do that, you can use the script pipeline.py which is inside the folder called "preprocess". In this way, you will transform you data in ordert to be feasible for the models.
```
!python preprocess/pipeline.py --data <DATASET> --language <language> --max_len <maximum length> --stage <stage_you_want_to_start_from> --jit <True if you want to use jit>
```
Note: if the parameter --jit is not specified, the code is run using it.


## Train extractor
Next step is to train extractor. To do that run the following cell:
```
!python train_extractor_ml.py --data <DATASET> --language <language> --lstm_hidden <lstm hidden layers> --batch <batch size> --ckpt_freq <checkpoint frequency> --max_word <words in a sentence are cut according to this parameter> --max_sent <sentences in a document are cut according to this parameter>
```
## Train abstractor
If you want to train abstractor, following line needs to be executed:
```
!python train_abstractor.py --data <DATASET> --language <language> --batch <batch size> --ckpt_freq <checkpoint_frequnce> --n_layer <number of layers> --n_hidden <number of hidden layers>
```
## Train RL model

Last, but not the least, model to be trained is the Reinforcement Learning's agent. To do that, run the following:
```
!python train_full_rl.py --data <DATASET> --language <language> --batch <batch size>
```
## Evaluate the model

In the end, evaluate the model using the script decode_full_model.py:
```
!python decode_full_model.py --data <DATASET> --language <language> --batch <batch size>
```

### Results
You should get the following results

Validation set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (41.01, 18.20, 38.57)  |  21.10 |
| + rerank           | (41.74, 18.39, 39.40)  |  20.45 |
| **new** |
| rnn-ext + abs + RL | (41.23, 18.45, 38.71)  |  21.14 |
| + rerank           | (42.06, 18.80, 39.68)  |  20.58 |

Test set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (40.03, 17.61, 37.58)  |  21.00 |
| + rerank           | (40.88, 17.81, 38.53)  |  20.38 |
| **new** |
| rnn-ext + abs + RL | (40.41, 17.92, 37.87)  |  21.13 |
| + rerank           | (41.20, 18.18, 38.79)  |  20.56 |

