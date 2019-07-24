# multimodal-listener

## experimentation

### TO-RUN
- [x] ~5 configs with 2 sources, LSTM, euc (190714, lr 3e-4 since epoch 272) (0: 282, 1: 17)~
- [x] ~5 configs with 2 sources, LSTM, euc (190713, lr 1e-5 since epoch 585) (0: 609)~
- [x] ~10 configs with 2 sources, LSTM, correlation~
- [x] 10 configs with 2 sources, VTF, euc (190714, lr 1e-4 since epoch 269) (0: 286)
- [ ] 10 configs with 2 sources, VTF, correlation 
- [x] ~10 configs with 2 sources, GAB, euc (190714, lr 1e-5 since epoch 231)~
- [ ] 10 configs with 2 sources, GAB, correlation 
- [x] ~10 configs with 2 sources, DRNN, euclidean~
- [x] ~10 configs with 2 sources, SRNN, euclidean~
- [x] 5 configs with 2 sources, LSTM, euclidean, MinLoss
- [x] 5 configs with 2 sources, VTF, euclidean, MinLoss

- [x] 10 configs with 3 sources, LSTM, euc
- [ ] 10 configs with 3 sources, LSTM, correlation 
- [x] 10 configs with 3 sources, VTF, euc (190714, lr 1e-4 since epoch 144)
- [ ] 10 configs with 3 sources, VTF, correlation 
- [ ] 10 configs with 3 sources, GAB, euc (initiated)
- [ ] 10 configs with 3 sources, GAB, correlation 

- [x] 10 configs with 5 sources, LSTM, euc
- [ ] 10 configs with 5 sources, LSTM, correlation 
- [x] 10 configs with 5 sources, VTF, euc (1/10: train loss: 11.32, val loss: 11.38, epoch: 23)
- [ ] 10 configs with 5 sources, VTF, correlation 
- [x] 10 configs with 5 sources, GAB, euc
- [ ] 10 configs with 5 sources, GAB, correlation 
## methodology

### data acquisition

### data processing

## development log

### issues

- always changing the global variables, making resuming a previous checkpoint with different global variables untrackable
- several documents recording results lying around with confusing file names
- few codes are modularized, with vast majority being monolithic
- confusing flag names
- need to make small tweaking easy


### TO-DOs

- have an idea of current results
- tidy up directories, make sure
  - resuming is working
  - things are correctly saved
- factorize
- make sure resuming is working
- compare performance of VTF on log-RMSE and pure MSE to see legitimacy of log
