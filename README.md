# G-MetaHiC
G-MetaHiC: A granular-ball metacell deep learning framework for predicting Hi-C contact maps from single-cell ATAC-seq data

# Overview
G-MetaHiC is a granular-ball metacell based deep learning framework for predicting Hi-C contact maps from single-cell ATAC-seq data. It integrated:
- a granular-ball–based metacell construction strategy
- a pseudo-bulk encoder module based on Transformer architecture with rotary positional embeddings to capture long-range dependencies in pseudo-bulk scATAC-seq signals
- a metacell encoder module based on graph convolutional networks (GCNs) to learn graph representations among metacells
- a feature fusion module based on cross multi-head attention mechanisms

## Framework of G-MetaHiC
<img width="1020" height="720" alt="Fig1_modified_version2" src="https://github.com/user-attachments/assets/a3668c1e-ede9-4b62-b71d-4a3e61379754" />


> 
>**Abstract**
>
>```
>The three-dimensional (3D) organization of chromatin plays a pivotal role in gene regulation, cell fate determination, and disease mechanisms.
>Although Hi-C enables genome-wide profiling of chromatin interactions, its utility is limited by high experimental cost, restricted resolution,
>and limited ability to capture cellular heterogeneity. To overcome these limitations, we propose G-MetaHiC, a deep learning framework that predicts
>high-resolution Hi-C contact maps from single-cell ATAC-seq data augmented with CTCF information. G-MetaHiC integrates three key innovations: (i) a
>granular-ball–based strategy for constructing metacells that effectively aggregates sparse single-cell accessibility signals; (ii) the first application,
>to our knowledge, of graph convolutional networks (GCNs) for ATAC/CTCF-guided 3D genome prediction, enabling the model to learn latent regulatory
>relationships; and (iii) a multimodal architecture that jointly models pseudo-bulk chromatin accessibility and metacell-level features, fusing these
>representations through cross-modal attention. Trained on human cell-line datasets and evaluated for cross-species generalization to mouse, G-MetaHiC
>predicts 10-kb bulk Hi-C contact matrices and consistently outperforms state-of-the-art baselines in contact-frequency prediction. Together, these results
>highlight a promising computational avenue for inferring 3D chromatin architecture from single-cell epigenomic data.
>```

# Installation
```
git clone https://github.com/HaoWuLab-Bioinformatics/G-MetaHiC

conda env update -f gmetahic.yml
```

# Usage
G-MetaHiC requires inputting single-cell ATAC-seq data and CTCF motif score
## Training
### Training on 3 cell types
```
python ./chromafold/train.py --data-path ./datasets/ -ct gm12878 HUVEC imr90
```
### Training on 1 cell types
```
python ./chromafold/train.py --data-path ./datasets/ -ct imr90
```
## Inference
```
python ./chromafold/inference.py --data-path ./datasets/ -ct imr90 --model-path ./checkpoints/chromafold_CTCFmotif.pth.tar -chrom 5
```


