[TOC]



# multimodal-listener

## roadmap

### plan

#### Jun
6.18
implement baseline in 
6.19
* MEETING

6.20

#### Jul

#### Aug 


## dataset
FSDKaggle 2018, based on 



## raw data

### toy dataset

|        ambient (noise)         |           active (voice)            |
| :----------------------------: | :---------------------------------: |
| crowd (100/108 train, 15 test) |    bird (100/152 train, 15 test)    |
|       *idling* (15 test)       |    gun (100/143 train, 15 test)     |
|                                | *coin_dropping* (15/51 test, 1 bad) |



### complete dataset

#### sources

##### active (human)

* openslr (variety) http://www.openslr.org/resources.php
* common voice (12 GB) https://voice.mozilla.org/en/datasets
* voxforge (small, fragmented) http://www.voxforge.org/home/listen

##### ambient

* environmental sounds http://www.cs.tut.fi/~heittolt/datasets
* AudioSet
* urbansound(nice template for mixed audio clips) https://urbansounddataset.weebly.com/urbansound.html



### processed data

#### combinations

holding the ground truths, we generate a rich collection of compositions that contains multiple sound sources. 



## methodology

### data acquisition

#### download

```bash
cd ~/audioset/download/
./download_all.sh
```

this script automatically parses the first 100 classes of sounds in `balances_train_segments.csv` from AudioSet and fetch the audio clips from Youtube to corresponding folders.

outputs:

* `out_path/` : clips 

### data processing

#### generating combinations

```bash
cd ~/multimodal_listener/data
python mix.py --num_sources [num_sources] --duration [duration] --out_path [out path] [--selected_classes [pre-selected categories]]
```

generates a random mixture of `num_class` sources, with each source occuring once in the mixture. the sources are drawn from a uniform distribution across all classes.

arguments:

* `--num_sources` : number of categories to mix. it should be between 2 and 100
* `--duration` : duration of the generated mixture in seconds
* `--out_path` : path that stores the outputs
* `--selected_classes` : optional. list of pre-selected classes

outputs:

* `abcd.wav` : result of mixing

* `abcd.csv` : metadata of mixture. start time, end time, class id for each source in the mixture
* `abcd/` :
  * `wav` files for ground truths in the mixture

example

```bash
python mix.py -n 3 -t 10 -o out/
```



#### pre-processing

create visual representation (spectrograms) for audio waves





### model design

#### problems to tackle

* generate individual tracks
  * ...
* great clarity
* ...
* quetions
  * why CNN? it's useful for capturing features, not spatial relationships
  * take the min during loss calculation - all output converge to one ground truth?
  * autoencoders?
  * 

#### baseline models

##### goals

the main goal of experimenting with baseline models is to explore 



#### performance models





### training

TBD

model

* dimensions - hyperparameters
* picking min (set-to-set mapping, not positional)



doing randomized search (fix categories, controlled)

* detect shift 
* not at generalizing stage yet





questions

* how did LSTM learn sequential patterns?
