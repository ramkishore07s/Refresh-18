# Ranking Sentences for Extractive Summarisation using Reinforcement Learning

This is a re-implementation of the paper <a href='http://www.aclweb.org/anthology/N18-1158'>Ranking Sentences for Extractive Summarisation using Reinforcement Learning</a> in PyTorch.
Find the original code <a href='https://github.com/EdinburghNLP/Refresh'>here</a>.

## Running the summarizer:

**NOTE: to run on CPU, set `--gpu_no` argument to -1**

* Download model weights from [here](https://drive.google.com/open?id=1uXWa4g5PZtGCICEtOm0ZxjpLe_sJiWK7) and save it.
* Download Glove cache from [here](https://drive.google.com/file/d/1MsijgP0oreEJwBM7QZPQRhZUe9-MTE-T/view?usp=sharing) and save it.
* run :
``` 
cd notebooks

ipython run.py -- --input_folder=<path/to/document_folder> --output_folder=<path/to/summary_folder> --weights=<path/to/saved/weights/> --glove=<path/to/glove/cache> --gpu_no=0 --summary_len=3
```

## Training

* Download dataset (neuralsum.zip) from [here](https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download).
* Run `preprocess.sh <path/to/neuralsum.zip>` to process data.
* - [ ] Script for training
