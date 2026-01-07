#hicdc+ for normalization
#
# Usage
# screen
# bsub -n 2 -W 10:00 -R 'span[hosts=1] rusage[mem=128]' -Is /bin/bash
# source /miniconda3/etc/profile.d/conda.sh
# conda activate chromafold_env
# cd ./hic_normalization
#
# Rscript ./hicdcplus/step1_hicdcplus_normalization_run.R \
# imr90.hic \
# 10000 \
# 'hg38'

# Rscript /home/user_home/zhouxiangfei/chengxianjin/ChromaFold-main/process_input/hic_normalization/hicdcplus/step1_hicdcplus_normalization_run.R \
# "/home/user_home/zhouxiangfei/chengxianjin/ChromaFold-main/chromafold/datasets/hic/ENCFF843MZF.hic" \
# 10000 \
# "hg38"

library(HiCDCPlus)
options(expressions = 500000)
memory.limit(size = 256000)
options(timeout = 6000)
.download_juicer <-
    function()
{
    fileURL <- "https://github.com/aidenlab/Juicebox/releases/download/v.2.13.07/juicer_tools.jar"

    bfc <- .get_cache()
    rid <- BiocFileCache::bfcquery(bfc, "juicer_tools_v2", "rname")$rid
    if (!length(rid)) {
     message( "Downloading Juicer Tools" )
     rid <- names(BiocFileCache::bfcadd(bfc, "juicer_tools_v2", fileURL ))
    }
    if (!isFALSE(BiocFileCache::bfcneedsupdate(bfc, rid)))
    BiocFileCache::bfcdownload(bfc, rid)

    BiocFileCache::bfcrpath(bfc, rids = rid)
    }

args <- commandArgs(trailing = TRUE)

#####################################
#      Step 0. Initial set-ups      #
#####################################

#1. Initial set-up
hicfile_path <- args[1] #location of the raw .hic file to be normalized
resolution <- as.integer(args[2]) #resolution of Hi-C
assembly <- args[3]

#2. Variable set-ups

if (assembly %in% c("mm9", "mm10")) {
  species <- "Mmusculus"
  chrom_length <- 19
} else if (assembly %in% c("hg19", "hg38")) {
  species <- "Hsapiens"
  chrom_length <- 22
} else {
  species <- "Unknown"
  print("Species not match")
}

outdir <- "~/chengxianjin/features/"
outpth <- "~/chengxianjin/intermediate/"

########################################
#      Step 1. Construct features      #
########################################

#Step 1. construct features (resolution, chrom specific)
chromosomes <- paste("chr", seq(1, chrom_length, 1), sep = "")
chromosomes[[chrom_length+1]] <- "chrX"
construct_features(output_path = paste0(outdir, assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC", sep = ""), # nolint
gen = species, gen_ver = assembly, sig = c("GATC", "GANTC"),
bin_type = "Bins-uniform", binsize = resolution, chrs = chromosomes)

#Step 2. generate gi_list instance

set.seed(1010)

#generate gi_list instance
gi_list <- generate_bintolen_gi_list(
  bintolen_path = paste0(outdir, "/", assembly,
  "_", as.integer(resolution / 1000), "kb_GATC_GANTC_bintolen.txt.gz"),
  gen = species, gen_ver = assembly)
#add .hic counts
gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list)) #nolint
head(gi_list$chr1)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
#############################################################
#      Step 2. Modeling and generate normalized values      #
#############################################################

# expand features for modeling
gi_list <- expand_1D_features(gi_list)
print("expand_1D_features done......")
#run HiC-DC+ on 2 cores
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)
head(gi_list$chr1)
##################################
#      Step 3. Save results      #
##################################
## -----------------------------------------------  ##
hicdc2hic_cheng <- function(gi_list, hicfile, mode = "normcounts", chrs = NULL, gen_ver = "hg19", memory=8) {
    options(scipen = 9999)
    #set memory limit to max if i386
    if (.Platform$OS.type=='windows'&Sys.getenv("R_ARCH")=="/i386") {
        gc(reset=TRUE,full=TRUE)
        utils::memory.limit(size=4095)
    }
    gi_list_validate(gi_list)
    binsize<-gi_list_binsize_detect(gi_list)
    if (is.null(chrs))
        chrs <- names(gi_list)
    tmpfile <- paste0(base::tempfile(), ".txt")
    gi_list_write(gi_list, tmpfile, columns = "minimal_plus_score", score = mode)
    ### 把标题行去掉
    file_content <- readLines(tmpfile)
    data_lines <- file_content[-1]  # 删除标题行
    writeLines(data_lines, tmpfile)

    #generate path to the file if not exists
    hicdc2hicoutput <- path.expand(hicfile)
    hicdc2hicoutputdir<-gsub("/[^/]+$", "",hicdc2hicoutput)
    if (hicdc2hicoutputdir==hicdc2hicoutput){
        hicdc2hicoutputdir<-gsub("\\[^\\]+$", "",hicdc2hicoutput)
    }
    if (hicdc2hicoutputdir==hicdc2hicoutput){
        hicdc2hicoutputdir<-gsub("\\\\[^\\\\]+$", "",hicdc2hicoutput)
    }
    if (!hicdc2hicoutputdir==hicdc2hicoutput&!dir.exists(hicdc2hicoutputdir)){
        dir.create(hicdc2hicoutputdir, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    }
    # run pre
    jarpath<-.download_juicer()
    ifelse(.Platform$OS.type=='windows'&Sys.getenv("R_ARCH")=="/i386",min(memory,2),memory)
    if (mode=="zvalue"){
        #make sure negative values get processed
        system2("java", args = c(paste0("-Xmx",as.character(memory),"g"), "-jar",
                                 path.expand(jarpath), "pre", "-v", "-d", "-n",
                                 "-r", binsize,
                                 "-m", -2147400000,
                                 path.expand(tmpfile), path.expand(hicdc2hicoutput),
                                 gen_ver))
    } else {
    system2("java", args = c(paste0("-Xmx",as.character(memory),"g"), "-jar",
                             path.expand(jarpath), "pre", "-v", "-d",
                             "-r", binsize, path.expand(tmpfile),
                             path.expand(hicdc2hicoutput),
                             gen_ver))
    }
    # remove file
    system2("rm", args = path.expand(tmpfile))
    return(hicdc2hicoutput)
}





# write normalized counts (observed/expected) to a .hic file
hicdc2hic(gi_list,
hicfile = paste0(outpth, "normalized.hic"),
          mode = "zvalue", gen_ver = assembly)

# 调用修改的hicdc2hic
# hicdc2hic_cheng(gi_list,
# hicfile = paste0(outpth, "normalized.hic"),
#           mode = "zvalue", gen_ver = assembly)

# write results to a text file
gi_list_write(gi_list,
fname = paste0(outpth, "normalized.txt.gz"))

for (chrom in names(gi_list)) {
  print(chrom)
  # Construct the file name
  file_name <- paste0(outpth, "normalized_", chrom, ".txt")
  # Extract the data for the current chromosome
  data <- gi_list[[chrom]]
  # Save the data to a text file
  write.table(data, file = file_name, sep = "\t",
  row.names = FALSE, col.names = TRUE, quote = FALSE)
}
