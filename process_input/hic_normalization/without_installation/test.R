library(BSgenome)
library(dplyr)
library(Rcpp)
library(base)
library(data.table)
library(dplyr)
library(tidyr)
library(GenomeInfoDb)
library(stats)
library(MASS)
options("scipen"=100, "digits"=4)

hic_path <- "/home/chengxianjin/projects/ChromaFold-main/chromafold/datasets/hic/HUVEC.hic"

straw_path <- "/home/chengxianjin/projects/ChromaFold-main/chromafold/datasets/hicdc/straw-R.cpp"
suppressWarnings({Rcpp::sourceCpp(path.expand(straw_path))})


chr_straw <- c('chr2:0:10000')
count_matrix <- straw(
        norm = 'NONE',
        fname = path.expand(hic_path),
        binsize = 10000,
        chr1loc = 'chr2',
        chr2loc = 'chr2',
        unit = 'BP')
print(count_matrix)