# PointNet Convolutional Neural Network Sample for 3D Pointcloud Classification

[PointNet][pointnet-paper] is a convolutional neural network architecture for applications concerning 3D recognition such as object classification and part segmentation. These sample codes implement a variant of PointNet for 3D object classification, for inference only with ModelNet10, providing a comprehensive example of using oneDNN. You can see the following initial instructions on using the samples.

## Obtain the Model Weights and Classes and Preparing an Input pointcloud

#### Python environment

To run the preprocessing step some python packages are necessary, install the following:

```bash
pip3 install numpy==2.2.3
pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Run preprocessing script

A preprocessing script is provided which unpacks the weights from a pre-trained pytorch model. The script also prepares an input pointcloud for testing inference. The pointcloud is made from 3D scans taken from the [ModelNet10][modelnet] dataset. The script requires an installation of [PyTorch][pytorch].

First, download the pre-trained PointNet weights and then move the pth file into the same directory containing the model.


```bash
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
rm ModelNet10.zip
python3 prepareData.py ModelNet10/ pointnet_model.pth
```

The weights will be saved to `data/` and the input pointcloud will be saved as `itemName_cloud.bin`.

## Build&Test on a pointcloud

To build this example perform the following steps:

#### 1. Setup oneAPI development environment

```bash
source ${INTEL_ONEAPI_INSTALL_FOLDER}/set_vars.sh
```

#### 2. Build the program using `cmake`

```bash
mkdir build
cd build
cmake ..
make
```

The sample executable is built as `pointnet`.

#### 3. Run the sample

You can get additional information during the execution of this sample by setting environment variable `DNNL_VERBOSE=1`.

To run the sample, provide as first argument the path to the directory containing the binary weight files and as second argument the path to the preprocessed point cloud file to be classified.

The expected output includes a classification index and (if enabled) timing measurements in nanoseconds.

```bash
pointnet ModelNet10/directory/extracted_data ModelNet10/directory/input_cloud/itemName_cloud.bin
```

Example output:

```bash
$ ./pointnet ~/Documents/git/pointnet/data ~/Documents/git/pointnet/sofa_cloud.bin
$ classed as 7 (i.e., sofa)
```


[pointnet-paper]: https://arxiv.org/pdf/1612.00593.pdf
[pytorch]: https://pytorch.org/
[modelnet]: https://modelnet.cs.princeton.edu/

