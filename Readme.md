# Ranking Sentences for Extractive Summarisation using Reinforcement Learning

This is a re-implementation of the paper <a href='http://www.aclweb.org/anthology/N18-1158'>Ranking Sentences for Extractive Summarisation using Reinforcement Learning</a> in PyTorch.
Find the original code <a href='https://github.com/EdinburghNLP/Refresh'>here</a>.

## Training

* Download dataset (neuralsum.zip) from [here](https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download).
* Run `preprocess.sh <path/to/neuralsum.zip>` to process data.
* Run `train.sh <type>` to train the model. \<type> can be either `cnn` or `dailymail` or `both`