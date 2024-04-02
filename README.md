# DynGRN
DynGRN is a tool for deciphering fine-grained dynamic life processes using time-series transcriptomics data. It takes time-series gene expression profiles (snapshots data or pseudo-time-series data) and cell-type-specific prior regulatory knowledge as inputs, then conducts cell-type-specific gene regulatory network (GRN) construction and dynamic GRNs rewiring. 
![overall_DynGRN_v8](https://github.com/lbright-liu/DynGRN/assets/96679804/fe1b1d21-668f-4c3d-b7c2-accc50197767)
The DynGRN model has the following benefits:
* Model time-series single-cell transcriptome data simultaneously from the two levels of network topology and temporal evolution
* Introduce cell-type-specific prior knowledge to guide model training more accurately. Even if specific prior knowledge is lacking, the integrated common prior gene interaction network can be used for pre-training and then further fine-tuning to construct GRN
* Widely used to analyze a variety of different fine-grained dynamic life processes

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
* pytorch 1.12.1
* Driver Version: 470.182.03
* CUDA Version: 11.4
* numpy, scipy, pandas, scikit-learn, tqdm, etc.

#### Setup a conda environment
```shell
conda create -y --name DynGRN python=3.8
conda activate DynGRN
```
#### Install using pip
Other packages can be easily installed by calling following command:
```shell
pip install xxx
```
### Prepare input files
The data for demo is in processed_data/mHSC-E. The sample data contained 1204 genes and 33 TFs at three time points, with a total of 1071 cells.
* **Target.csv**: all genes and their index numbers.
* **TF.csv**: all TFs and their index numbers in Target.csv.
* **label.csv**: cell-type-specific prior regulatory knowledge, collected from gene regulatory databases, biological experiments, and other gold standards.
* **Train_set1.csv, Validation_set1.csv, and Test_set1.csv**: label data for model training and evaluation, which can be obtained using the previous three files by running the following command:
  ```shell
  python train_test_split.py
  ```
* **mEc3_expression.csv**: time-series gene expression matrix for all time points, with rows representing genes and columns representing cells.

### Cell-type-specific GRN inference
Taking time-series single-cell gene expression matrix and cell type-specific prior regulatory knowledge as inputs, the following commands are executed for GRN construction:
```shell
python cell_type_specific_GRN_main.py
```
For some cell types lacking specific prior knowledge, GRNS can be constructed by transfer learning, which is pre-trained using integrated common prior gene interaction network (**demo_data/NicheNet**), and then fine-tuned using a small amount of cell-type-specific prior knowledge:
```shell
python GRN_TL_main.py
```
The result output is as follows:

```
|TF|Target|score|
|JUN|TLK1|0.756|
|REL|CBFB|0.831|
|SOX6|TEAD2|0.246|
...
```
### Dynamic GRNs reconstruction
We can reconstruct the directed or undirected GRN at each time point using time-series expression data.
#### Undirected dynamic network
```shell
python dgrn_main.py --flag False
```
#### Directed dynamic network
```shell
python dgrn_main.py --flag True
```
The resulting output is shown in the **'demo_data/hesc2/regulatory_tk.csv'** file, which takes the top 20% of the predicted scores.

### Identification of key genes based on dynamic network perturbation
Identification of key genes (TFs and Non-TFs) based on dynamic network (including gene-gene edges) perturbation. As shown in **Fig. e**, firstly, a gene in the dynamic network is knocked out, and then the perturbation score of the gene is obtained by calculating the change of network entropy before and after the knockout. Finally, key genes are identified based on perturbation score.
Taking the **'demo_data/hesc2/regulatory_tk.csv'** file as sample inputs, obtain the perturbation score for each gene by using the following command:
```shell
python dynamic_perturbation.py
```
### Detection of co-occurring transcriptional regulatory modules (TRMs)
Taking the "demo_data/hesc2/regulatory_tk.csv" ** file as sample input, obtain the transcriptional regulation modules (TRMs) that are always present throughout the dynamic process by using the following command:
```shell
python TRMs_detection.py
```
### K-means clustering for dynamic regulatory edges
By k-means clustering of score vectors composed of regulatory edges in dynamic GRNs, different edge clusters can be obtained, which is conducive to further study of fine-grained dynamic processes.
```shell
python regulatory_edge_cluster.py
```
### Reconstruction of stage-specific GRNs
DynGRN was extended to maize time-series transcriptome data to demonstrate its potential to model time-series bulk transcriptome data as well. The whole process is similar except that the integration of input data is different from that of single-cell processing.
![image](https://github.com/lbright-liu/DynGRN/assets/96679804/34c2b86a-ac1f-4238-adb9-79c5bb55648d)
Build stage-specific dynamic GRNs using the following command:
```shell
python Stage_Specific_GRN.py --input_file xxx_tk.csv
```
**'xxx_tk.csv'** can be time-series single-cell gene expression data or time-series bulk gene expression data.



