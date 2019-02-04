# multimodal-listener

### dataset

Using AudioSet from Google AI.

####toy dataset

ambient (noise)

* crowd (100/108 train, 15 test)
* *idling* (15 test)

active (voice)

* bird (100/152 train, 15 test)
* gun (100/143 train, 15 test)
* *coin_dropping* (15/51 test, 1 bad)



superimpose level

1. V-N: 1 active + 1 ambient

2. MV-N: [n] active + 1 ambient

3. MV-MN: [n] active + [n] ambient