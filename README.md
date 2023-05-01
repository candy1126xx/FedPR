# FedPR: A Method based on Posterior Distribution of Representation for Heterogeneous Federated Learning

## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
* Download train and test datasets manually from the given links, or they will use the defalt links in torchvision.
* Experiments are run on MNIST, Fashion MNIST, Federated Extended MNIST and CIFAR10.

http://yann.lecun.com/exdb/mnist/

https://s3.amazonaws.com/nist-srd/SD19/by_class.zip

http://www.cs.toronto.edu/∼kriz/cifar.html

## Running the experiments

* In the Model-Agnostic scenario, run FedPR:
```
python main.py --model_name mnistcnn4 --dataset mnist --n 3 --algorithm FedPR --gamma 1.0
```
* In the Resource-Multilevel scenario, run pFedPR+:
```
python main.py --model_name mnistcnn4 --dataset mnist --n 3 --algorithm pFedPR+ --gamma 1.0
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.


## Options

* ```--model_name:```  Local model name. Default: 'mnistcnn4'. Options: 'mnistcnn4', 'cifarcnn4', 'resnet18'
* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fashion', 'femnist', 'cifar10'
* ```--n:```     Pathological: num of shards per user. Default: 2.
* ```--seed:```     Random Seed. Default set to 114.
* ```--algorithm:```       Default: 'Local'.
* ```--gamma:```       A regularization parameter. Default: 0.
* ```--num_glob_iters:```       Num of communication rounds. Default: 100
* ```--num_users:```  Num of clients. Default: 100

The more various paramters parsed to the experiment are given in ```main.py```.