# Pruning with Tensorflow

This repository was created to give a demonstration on how to apply pruning to a neural network.

Pruning is a very simple compression strategy. It's comprised of the following steps:

1. Train a neural network as usual.
2. Replace with zeros all the weights in the network that have absolute values below a certain (predefined) threshold.
3. Fine-tune on a pruned network (train for 10-20 epochs with low learning rate).
4. Replace all the matrices in a network with sparse matrices.

This leads to a compression rate (in terms of number of bytes) in range of 5 to 20 depending on a network type.


I trained a simple fully-connected model on MNIST dataset for the purpose of demonstration.
I was able to compress my model in almost 10 times.


To recreate my results, please do the instructions listed below.

**Package installation**

I'd highly recommend to use [virtualenv](https://virtualenv.pypa.io/en/stable/) to isolate this project from other python projects.
However you could install all the packages into the system if you want to.

NOTE: I used ```python3.5``` for this project.

To install the packages, run the following terminal command:

```
pip install -r requirements.txt
```

**Experiments**

Run 

```
python train_network_dense.py
```

to train a regular dense model. MNIST dataset is downloaded automatically (it may take some time).

Weights distribution after this phase might look something like this:

<img src="https://github.com/ex4sperans/pruning_with_tensorflow/blob/master/pics/weights_distribution_before_pruning.png?raw=true" width="700"/>

Then, you need to prune a network and fine-tune it.

Run 

```
python prune_network.py
```

This script creates a new pruned model, fine-tunes it, and generates images with weights distribution (before pruning, after pruning and after fine-tuning).

After removing the small weights, distribution is like this (zero weights are excluded):

<img src="https://github.com/ex4sperans/pruning_with_tensorflow/blob/master/pics/weights_distribution_after_pruning.png?raw=true" width="700"/>

After fine-tuning the distribution looks much more smoothed (as some of the weights have changed to compensate the pruned ones):

<img src="https://github.com/ex4sperans/pruning_with_tensorflow/blob/master/pics/weights_distribution_after_fine_tuning.png?raw=true" width="700"/>

Finally, it's a time to create a sparse network and deploy it.

Run 

```
python deploy_pruned_model.py
```

to do it.

Last script deploys sparse model, measures accuracy and saves created sparse weights.

Alternatively, you could run all at once:

```
./run_experiments.sh
```

**Additional info**

Configs for all models (dense, dense pruned, and sparse) could be found in ```configs.py```

You could run tests with the following command:

```
./run_tests.sh
```
