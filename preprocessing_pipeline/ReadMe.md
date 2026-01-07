## Data Preprocessing for G-MetaHiC 

All processing steps have been summarized in [`fragment_to_input.sh`](https://drive.google.com/drive/folders/1s5FCXnRsgT4PzJgi16Ak3IFFE2B7Rp9Z) file. Below is a more detailed explanation of the steps. 

### **Step 0**. Filter and merge fragement files

- **Input:** single-cell ATAC-seq fragment file and barcode file. Example data: [google drive](https://drive.google.com/drive/folders/1s5FCXnRsgT4PzJgi16Ak3IFFE2B7Rp9Z)


Filter and merge fragments of a specific cell type from multiple fragment datasets. In the example data from google drive, since we only have a single scATAC-seq fragment dataset, we will skip this step. 

### **Step 1**. Run [ArchR](https://drive.google.com/drive/folders/1s5FCXnRsgT4PzJgi16Ak3IFFE2B7Rp9Z) to calculate co-accessibility

- **Input:** scATAC-seq fragment file from [google drive](https://drive.google.com/drive/folders/1s5FCXnRsgT4PzJgi16Ak3IFFE2B7Rp9Z)
- **Packages:** [ArchR](https://www.archrproject.com/bookdown/co-accessibility-with-archr.html) for creating co-accessibility matrix, [Samtools](https://www.htslib.org/) and [bedtools2](https://github.com/arq5x/bedtools2) for converting data format.
- **Script:** [`ArchR_preparation.R`](https://github.com/HaoWuLab-Bioinformatics/G-MetaHiC/blob/master/preprocessing_pipeline/ArchR_preparation.R)

#### **Define arguments**
```
SCRIPT_DIR="/gmetahic/preprocessing_pipeline"
SAVE_LOC="/home/data"
DATA_PREFIX="imr90_fragments"
FRAG_LOC="/home/data/fragments_by_celltype"
FRAG_FILE_PREFIX="${DATA_PREFIX}"
GENOME_ASSEMBLY="hg38"
```
#### **Convert fragments.tsv.gz file into sorted bgzip format**

- Without `bedtools2` installed: 
```
mkdir -p "${SAVE_LOC}"/archr_data/"${DATA_PREFIX}"
ARCHR_LOC="${SAVE_LOC}"/archr_data/"${DATA_PREFIX}"

# export PATH=/home/packages/samtools/bin:$PATH
# export PATH=/home/packages/samtools/samtools/bin:$PATH

# use linux sort
gunzip -c "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}.tsv.gz" | sort -k 1,1 -k2,2n > "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv"
htsfile "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv" # check whether file is in bed format
bgzip "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv" # bgzip file for tabix

rm "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv.gz.tbi" # remove previously calculated .tbi file
```

- Alternatively, with `bedtools2` installed: 
```
mkdir -p "${SAVE_LOC}"/archr_data/"${DATA_PREFIX}"
ARCHR_LOC="${SAVE_LOC}"/archr_data/"${DATA_PREFIX}"

# export PATH=/home/packages/samtools/bin:$PATH
# export PATH=/home/packages/samtools/samtools/bin:$PATH
# export PATH=/home/packages/bedtools2/bin:$PATH

gunzip -c "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}.tsv.gz" | bgzip  > "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz.tsv.gz" # convert gzip to bgzip
sortBed -i "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz.tsv.gz" > "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv" # sort merged bgzip file
htsfile "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv" # check whether file is in bed format
bgzip "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv" # bgzip file for tabix

rm "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv.gz.tbi" # remove previously calculated .tbi file
rm "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz.tsv.gz" # remove intermediate files
```
#### Create folders to store gmetahic input
```
cd "${SAVE_LOC}"
mkdir -p atac
mkdir -p dna
mkdir -p predictions
```
#### Copy CTCF motif score to the designated folder 

CTCF motif data for each genome assembly can be found at [google drive](https://drive.google.com/drive/folders/16zgPAc1TGMWvkyouswQpvZcVqo7_HKla)
```
cp /home/data/dna/* "${SAVE_LOC}"/dna/
```
#### Run R to create [LSI](https://www.archrproject.com/bookdown/iterative-latent-semantic-indexing-lsi.html) file using ArchR 
```
Rscript "${SCRIPT_DIR}"/ArchR_preparation.R \
"${DATA_PREFIX}" \
"${ARCHR_LOC}" \
"${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv.gz" \
"${GENOME_ASSEMBLY}"
```
### **Step 2**. Calculate tile files using [ArchR](https://www.archrproject.com/bookdown/co-accessibility-with-archr.html)

- **Input:** sorted bgzip fragment file created in previous step, LSI file created in previous step, and barcode file (example barcode file can be found in [google drive](https://drive.google.com/drive/folders/1s5FCXnRsgT4PzJgi16Ak3IFFE2B7Rp9Z)
- **Script:** [`scATAC_preparation.py`](https://github.com/HaoWuLab-Bioinformatics/G-MetaHiC/blob/master/preprocessing_pipeline/scATAC_preparation.py)
```
python "${SCRIPT_DIR}"/scATAC_preparation.py \
--cell_type_prefix "${DATA_PREFIX}" \
--fragment_file  "${FRAG_LOC}"/"${FRAG_FILE_PREFIX}_bgz_sorted.tsv.gz" \
--barcode_file "${ARCHR_LOC}"/archr_filtered_barcode.csv \
--lsi_file "${ARCHR_LOC}"/archr_filtered_lsi.csv \
--genome_assembly "${GENOME_ASSEMBLY}" \
--save_path "${SAVE_LOC}"
```

**Complete!**

### **Final Data Check**

Until this step, we have all the input files ready for G-MetaHiC predictions. A complete data processing should have created 3 folders: 
- `atac`: including `imr90_tile_50bp_barcode.npy`,`imr90_tile_50bp_dict.p`,`imr90_tile_500bp_barcode.npy`,`imr90_tile_500bp_dict.p`,`imr90_tile_pbulk_50bp_dict.p`, `metacell_mask.csv`
- `ctcf`: including CTCF motif data, `hg38_ctcf_motif_score.p`
- `prediction`: where G-MetaHiC predictions will be saved in.

An example data folder can be found at [google drive](https://drive.google.com/drive/folders/1ot0u8GDWvku9_XS7Cxk_QyYUyBQrAM32?usp=sharing)
















