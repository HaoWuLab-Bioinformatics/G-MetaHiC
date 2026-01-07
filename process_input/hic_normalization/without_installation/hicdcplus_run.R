#!/usr/bin/env Rscript

# Run HiC-DC+ to generate z-score normalized Hi-C library.
#
# Usage
#
# Rscript ./gmetahic/process input/hic_normalization/hicdcplus_run.R \
# 10000 \
# "./gmetahic/datasets/hicdc/hg38_MboI_10kb_features.rds" \
# "./data/hic/imr90/hicdc/" \
# "Hsapiens" \
# "hg38" \
# "./data/hic/imr90/ENCFF281ILS.hic" \
# "./gmetahic/datasets/hicdc/straw.cpp" \
# "./gmetahic/datasets/hic/juicer_tools.1.9.9_jcuda.0.8.jar" \



source("~/process_input/hic_normalization/without_installation/hicdcplus_pipeline.R")
setwd("~/process_input/hic_normalization/without_installation/")

args <- commandArgs(trailing = TRUE)

#####################################
#      Step 0. Initial set-ups      #
#####################################

#1. Initial set-up
# binsize <- args[1] #resolution of Hi-C
binsize <- as.integer(args[1])
bintolen_path <- args[2] #location of the genomic features (downloaded from google drive) #nolint
output_path <- args[3] #location for where to save the normalized files
gen <- args[4] #species
gen_ver <- args[5] #sequencing assembly
input_hic_file <- args[6] #location of the raw .hic file to be normalized
straw_path <- args[7] #location of the downloaded straw.cpp file
juicer_jar <- args[8]

##########################################
#      Step 1. Prepare HiCDC object      #
##########################################
hic2hicdc(hic_path = input_hic_file,
          bintolen_path = bintolen_path,
          straw_path = straw_path,
          gen = gen, gen_ver = gen_ver, binsize = binsize,
          bin_type = "Bins-uniform",
          inter = FALSE, Dthreshold = 2e6,
          # chrs = sprintf("chr%s", c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          #     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "X")),
          chrs = sprintf("%s", c(1:22, "X")),
          # chrs = sprintf("chr%s", 2),
          # chrs = sprintf(c("chr2")),
          output_path = output_path)


#########################################
#      Step 2. HiCDC normalization      #
#########################################

chrom <- c("chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
        "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
        "chr13", "chr14", "chr15","chr16", "chr17", "chr18",
        "chr19", "chr20", "chr21", "chr22", "chrX")



# Calculate normalization for all chromosomes
for (chr in chrom){
  print(paste0(chr,"&&&&&&&&&&&&&&&&&&&&&&&&&&&"))
  hicdcplus_model(file=paste0(output_path,"hESC_", binsize/1000, "kb_", chr, "_matrix.txt"), #nolint
  bin_type = "Bins-uniform", gen = gen, gen_ver = gen_ver,
  covariates = c("gc", "map", "len"), distance_type = "spline",
  output_path = output_path, chrs = chr,
  df = 6, Dmin = 0, Dmax = 2e6, ssize = 0.01, model_file = FALSE)
}


print(paste0("################ norm 完毕 ################"))

########################################################
#      Step 3. Save normalized objects into files      #
########################################################

#1. Save z-score normalization (z-score = (observed-expected)/std)


hicdc2hic(input_path = output_path, jarfile = juicer_jar,
binsize = binsize, gen = gen, gen_ver = gen_ver,
mode = "zvalue",
# output_file = paste(output_path, "ENCFF281ILS_zvalue.hic", sep = ""),
output_file = paste(output_path, "hESC_zvalue.hic", sep = ""),
chrs = sprintf("chr%s", c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "X")))










