# l2 normalization layer based on TensorRT C++ API

This is a C++ implementation of the L2 Normalization of input features, using tensorrt api,
based on the code from [tensorrtx/alexnet](https://github.com/wang-xinyu/tensorrtx/tree/master/alexnet). Verified by tensorrt 7.
```shell

// build and run

mkdir build && cd build

cmake ..

make

sudo ./l2norm -s   // serialize model to plan file i.e. 'l2norm.engine'

sudo ./l2norm -d   // deserialize plan file and run inference

```