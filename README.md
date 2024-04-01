# DynGRN
DynGRN is a tool for deciphering fine-grained dynamic life processes using time-series transcriptomics data. It takes time-series gene expression profiles (snapshots data or pseudo-time-series data) and cell-type-specific prior regulatory knowledge as inputs, then conducts cell-type-specific gene regulatory network (GRN) construction and dynamic GRNs rewiring. 
![overall_DynGRN_v8](https://github.com/lbright-liu/DynGRN/assets/96679804/fe1b1d21-668f-4c3d-b7c2-accc50197767)
The DynGRN model has the following benefits:
* uses multi-task learning allowing the learning procedure to be informed by the shared infor- mation across cell
* uses multi-task learning allowing the learning procedure to be informed by the shared infor- mation across cel
* uses multi-task learning allowing the learning procedure to be informed by the shared infor- mation across cel

# Installation
### Download the repository
```shell
git clone git@github.com:lbright-liu/DynGRN
cd DynGRN
```
### Install required packages
We recommend using Anaconda to get the dependencies. If you don't already have Anaconda, install it by following the instructions at this link: https://docs.anaconda.com/anaconda/install/. DynGRN was originally tested on Ubuntu 18.04.6 LTS with Python (3.8~3.9), please use an NVIDIA GPU with CUDA support for GPU acceleration.
#### Requirements
* python 3.8
* pytorch xx
* torch-geometric: xx
* CUDA Version: 11.4

#### Setup a conda environment
```shell
conda create -y --name DynGRN python=3.xx
conda activate DynGRN
```
#### Install using pip
Other packages can be easily installed by calling following command:
```shell
pip install xxx
```
### Prepare input files

### Cell-type-specific GRN inference

### Dynamic GRNs reconstruction

#### Undirected dynamic network

#### Directed dynamic network

### Identification of key genes based on dynamic network perturbation

### Detection of co-occurring transcriptional regulatory modules (TRMs)

### K-means clustering for dynamic regulatory edges


### Reconstruction of stage-specific GRNs
