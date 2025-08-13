<img width="1424" height="1215" alt="image" src="https://github.com/user-attachments/assets/f62634af-7b5c-4b9e-9eff-3a5c1a3e2c37" /># scDGRN (Running tutorials are continuously updated...)
scDGRN is a tool for deciphering fine-grained dynamic life processes using time-series transcriptomics data. It takes time-series gene expression profiles (snapshots data or pseudo-time-series data) and cell-lineage-specific prior regulatory knowledge as inputs, then conducts cell-lineage-specific gene regulatory network (GRN) construction and dynamic GRNs rewiring. 


![image](https://github.com/user-attachments/assets/a947bb09-eb34-4cd4-ad5d-f1b0554f1f22)


**The scDGRN model has the following benefits:**
* Model time-series single-cell transcriptome data simultaneously from the two levels of network topology and temporal evolution
* Introduce cell-type-specific prior knowledge to guide model training more accurately. Even if specific prior knowledge is lacking, the integrated common prior gene interaction network can be used for pre-training and then further fine-tuning to construct GRN
* Widely used to analyze a variety of different fine-grained dynamic life processes

## Installation
### Download the repository
```shell
git clone https://github.com/lbright-liu/scDGRN
cd scDGRN
```
### Install required packages
We recommend using Anaconda to get the dependencies. If you don't already have Anaconda, install it by following the instructions at this link: https://docs.anaconda.com/anaconda/install/. scDGRN was originally tested on Ubuntu 18.04.6 LTS with Python (3.8~3.9), please use an NVIDIA GPU with CUDA support for GPU acceleration.
#### Requirements
* python 3.8
* pytorch 1.12.1
* Driver Version: 470.182.03
* CUDA Version: 11.4
* numpy, scipy, pandas, scikit-learn, tqdm, etc.

#### Setup a conda environment
```shell
conda create -y --name scDGRN python=3.8
conda activate scDGRN
```
#### Install using pip
Other packages can be easily installed by calling following command:
```shell
pip install torch==1.12.1
pip install xxx
```
## Prepare input files
The data for demo is in processed_data/mHSC-E. The sample data contained 1204 genes and 33 TFs at three time points, with a total of 1071 cells.
* **Target.csv**: all genes and their index numbers.
* **TF.csv**: all TFs and their index numbers in Target.csv.
* **label.csv**: cell-type-specific prior regulatory knowledge, collected from gene regulatory databases, biological experiments, and other gold standards.
* **Train_set1.csv, Validation_set1.csv, and Test_set1.csv**: label data for model training and evaluation, which can be obtained using the previous three files by running the following command:
  ```shell
  python train_test_split.py
  ```
* **mEc3_expression.csv**: time-series gene expression matrix for all time points, with rows representing genes and columns representing cells.

## Cell-lineage-specific GRN inference
Taking time-series single-cell gene expression matrix and cell type-specific prior regulatory knowledge as inputs, the following commands are executed for GRN construction (We have prepared sample data for the mEc3 cell line, and you can directly run the following command. The codes related to user data upload and GRN construction are being continuously updated...):
```shell
python cell_lineage_specific_GRN_main.py
```
For some cell types lacking specific prior knowledge, GRNs can be constructed by transfer learning, which is pre-trained using integrated common prior gene interaction network (**demo_data/NicheNet**), and then fine-tuned using a small amount of cell-type-specific prior knowledge (We have prepared sample data for the mDC cell line, and you can directly run the following command. The codes related to user data upload and dynamic GRNs reconstruction are being continuously updated...):
```shell
python GRN_TL_main.py
```
The result output is as follows:

```
TF      Target    score
JUN     TLK1      0.756
REL     CBFB      0.831
SOX6    TEAD2     0.246
...
```
## Dynamic GRNs reconstruction
We can reconstruct the dynamic GRNs at each time point using time-series expression data (take the hesc2 dataset as an example).  
**Traing model:**
```shell
python dgrn_main.py
```
**Reconstructing:**
```shell
python output_TG_network.py
```
The resulting output is shown in the **'processed_data/hesc2/hesc2_regulatory_tk.csv'** file, which takes the top 20% of the predicted scores.

## Identification of key genes based on dynamic network perturbation
Identification of key genes (TFs and Non-TFs) based on dynamic network (including gene-gene edges) perturbation. As shown in **Fig. d**, firstly, a gene in the dynamic network is knocked out, and then the perturbation score of the gene is obtained by calculating the change of network entropy before and after the knockout. Finally, key genes are identified based on perturbation score.
Taking the **'processed_data/hesc2/hesc2_regulatory_tk.csv'** file as sample inputs, obtain the perturbation score for each gene by using the following command:
```shell
python dynamic_perturbation.py
```

## Mining of dynamic regulatory patterns
Based on the inferred dynamic gene regulatory network, we can obtain the TF groups of different dynamic regulatory patterns by using the following command:
```shell
python dynamic_pattern.py
```

## Discovery of co-regulons
Based on the inferred dynamic gene regulatory network, we can observe the phenomenon of multi-factor coordinated regulation throughout the entire dynamic process:
```shell
python extract_co_regulons.py
```

## Reconstruction of stage-specific GRNs
scDGRN is capable of constructing stage-specific GRN, which is of vital importance for depicting the fine-grained developmental state of cells.
<!--
![image](https://github.com/lbright-liu/scDGRN/assets/96679804/34c2b86a-ac1f-4238-adb9-79c5bb55648d)
-->
Build stage-specific dynamic GRNs using the following command:
```shell
python Stage_Specific_GRN.py --input_file xxx_tk.csv
```
**'xxx_tk.csv'** can be time-series single-cell gene expression data or time-series bulk gene expression data.



