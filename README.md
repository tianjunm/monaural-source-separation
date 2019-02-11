[TOC]



# multimodal-listener

##dataset

Audio samples are mainly gathered from Google AI's AudioSet.

Toy dataset is consist of 5 categories. There are 100 categories in total in the complete dataset.



### raw data

####toy dataset

|        ambient (noise)         |           active (voice)            |
| :----------------------------: | :---------------------------------: |
| crowd (100/108 train, 15 test) |    bird (100/152 train, 15 test)    |
|       *idling* (15 test)       |    gun (100/143 train, 15 test)     |
|                                | *coin_dropping* (15/51 test, 1 bad) |



####complete dataset

#####human

* openslr (variety) http://www.openslr.org/resources.php
* common voice (12 GB) https://voice.mozilla.org/en/datasets
* voxforge (small, fragmented) http://www.voxforge.org/home/listen

##### environment

* environmental sounds http://www.cs.tut.fi/~heittolt/datasets
* AudioSet
* urbansound(nice template for mixed audio clips) https://urbansounddataset.weebly.com/urbansound.html



###processed data

#### combination

holding the ground truths, we generate a rich collection of compositions that contains multiple sound sources.



##usage

### data acquisition

####download

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
python mix.py [-n num_class] [-t length] [-o out_path]
```

generates a random mixture of `num_class` sources, with each source occuring once in the mixture. the sources are drawn from a uniform distribution across all classes.

arguments:

* `-n` : number of categories to mix. it should be between 2 and 100
* `-t` : duration of the generated mixture in seconds
* `-o` : path that stores the outputs

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

stft for spectrogram



### training

TBD





