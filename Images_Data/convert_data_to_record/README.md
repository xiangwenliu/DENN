#Make tfrecord file

## CIFAR10
* python cifar10_convert_to_record.py
* download cifar10 dataset to path "./data/cifar10/" and convert data to tfrecord data
* split 2000 samples from training data as validation data

## CIFAR100
* python cifar100_convert_to_record.py
* download cifar100 dataset to path "./data/cifar100/" and convert data to tfrecord data
* split 2000 samples from training data as validation data

## Fashion
* python fashion_convert_to_records.py
* download Fashion dataset to path "./data/fashion_mnist/" and convert data to tfrecord data
* split 5000 samples from training data as validation data

## MNIST
* python  mnist_convert_to_records.py
* download MNIST dataset to path "./data/MNIST_data/" and convert data to tfrecord data
* split 5000 samples from training data as validation data
