#  Monaural Audio Source Separation

## dev log
### comparing two schemes of storing dataset
current implementation of the datasets takes up tremendous amount of space. they also take a really long time to generate due to tens of thousands of system I/Os. the following experimentation is conducted to determine whether the 'view' oriented design will be a better way of dataset storage for future experiments. it will be very flexible, and will greatly reduce the amount of space needed if the doesn't increase the dataloader overhead too much.
#### original method
SignalDataset
```
for batch in dataset:
  numpy loads npy file with a spectrogram #(mixture, gt * num_sources)
  torch reshapes the loaded numpy array 
```

#### storage using 'view'
```
for batch in dataset:
  load(wav_files) # (gt * num_sources)
  AudioSegment manipulation 
  scipy.stft(manipulated wav file)
  torch reshape the spectrogram array
  
```


new scheme uses pytorch implementation of stft.

### issues

- the strange delay of Tensor(list)
- always changing the global variables, making resuming a previous checkpoint with different global variables untrackable
- several documents recording results lying around with confusing file names
- few codes are modularized, with vast majority being monolithic
- confusing flag names
- need to make small tweaking easy



### TO-DOs
- make sure the following steps are safe after modification:
  - [x] data generation
  - [x] dataloader
  - [x] training with correct mixture & ground truths
  - [x] istft
  - [x] benchmark calculation

- specify what the datasets should look like
  - train, val, test
  - all three contain the same set of categories
  - each category contains `file_selection_range` number of files within each category to choose from
  - each training instance:
    - selected filenames from different categories
    - train/val/test spec determines what files within certain category can be selected
    - a. start, end on their corresponding original files
    - b. start, end of them in the mixture
    - c. category name (for reference convenience)
  - store as a csv file 
    - filename#1, filename#2, mstart#1, mend#1, mstart 
   
- design & implement mix.py
  - keep Mixer in case new scheme fails

- modify dataset.py
  - loading training instances
  - `__getitem__` loads ONE datapoint
    - load num_source rows of filename.wav
    - a. for extracting from filename.wav
    - b. for padding it to ground_truths
    - combine to get mixture
    - apply stft to mixture, ground_truths
    - return torch tensors

- apply changes to aws
- test on aws
- train

- tidy up directories, make sure
  - [ ] resuming is working
  - [ ] things are correctly saved
- factorize
  - [ ] resuming is working
  - [ ] things are correctly saved
