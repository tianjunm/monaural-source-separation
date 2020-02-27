#  Monaural Audio Source Separation
This is a Pytorch implementation of [WildMix Dataset and Spectro-Temporal Transformer Model for Monoaural Audio Source Separation](https://arxiv.org/abs/1911.09783) that performs dataset creation, model training, and audio source separation.

## Requirements

## Dataset
### Download the Raw Data

### Configure the Mixture Dataset

### Use the Mixture Dataset

## Training the Network

## Performing Source Separation

## Log
### updates

### issues
- [ ] the strange delay of Tensor(list)
- [ ] always changing the global variables, making resuming a previous checkpoint with different global variables untrackable
- [ ] several documents recording results lying around with confusing file names
- [ ] few codes are modularized, with vast majority being monolithic
- [ ] confusing flag names
- [ ] mmsdk is not available on conda


