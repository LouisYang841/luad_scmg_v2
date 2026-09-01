#!/usr/bin/env Rscript
# Run SCEVAN independently on one sample's raw UMI matrix.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop("usage: 74_run_scevan.R INPUT_DIR OUTPUT_DIR [CORES]")
}
input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
cores <- if (length(args) == 3) as.integer(args[[3]]) else 8L
sample_id <- basename(input_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)
setwd(output_dir)
scevan_output <- "output"
dir.create(scevan_output, showWarnings = FALSE)

library(Matrix)
library(SCEVAN)

message("Reading raw sparse matrix for ", sample_id)
counts <- readMM(gzfile(file.path(input_dir, "matrix.mtx.gz")))
genes <- readLines(file.path(input_dir, "genes.tsv"))
barcodes <- readLines(file.path(input_dir, "barcodes.tsv"))
if (!identical(dim(counts), c(length(genes), length(barcodes)))) {
  stop("matrix dimensions do not match genes/barcodes")
}
rownames(counts) <- genes
colnames(counts) <- barcodes
if (length(counts@x) > 0 && (min(counts@x) < 0 || any(counts@x != round(counts@x)))) {
  stop("SCEVAN input is not non-negative integer raw counts")
}

set.seed(0)
started <- Sys.time()
prediction <- pipelineCNA(
  counts,
  sample = sample_id,
  par_cores = cores,
  norm_cell = NULL,
  SUBCLONES = FALSE,
  ClonalCN = FALSE,
  plotTree = FALSE,
  organism = "human",
  output_dir = scevan_output
)
prediction$barcode <- rownames(prediction)
prediction <- prediction[, c("barcode", setdiff(colnames(prediction), "barcode")), drop = FALSE]
write.csv(prediction, "predictions.csv", row.names = FALSE)

config <- list(
  sample_id = sample_id,
  n_genes_input = nrow(counts),
  n_cells_input = ncol(counts),
  cores = cores,
  normal_labels_supplied = FALSE,
  subclones = FALSE,
  clonal_cn = FALSE,
  scevan_version = as.character(packageVersion("SCEVAN")),
  elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs"))
)
jsonlite::write_json(config, "run_config.json", pretty = TRUE, auto_unbox = TRUE)
writeLines(capture.output(sessionInfo()), "session_info.txt")
message("Wrote ", file.path(output_dir, "predictions.csv"))
print(table(prediction$class, useNA = "ifany"))
