#!/usr/bin/env Rscript

# Extract only the exact author cell-to-meta_group mapping from df_cell_graph.rds.
# The RDS format itself is not stream-readable, so readRDS has one unavoidable
# in-memory load.  Everything after that keeps only the required compact columns
# and writes directly to a compressed table.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript extract_big_cell_graph_rds_for_fm.R INPUT.rds OUTPUT.tsv.gz")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_path <- args[[2]]

message(sprintf("[rds] reading %s", input_path))
object <- readRDS(input_path)
if (!is.data.frame(object)) {
  object <- as.data.frame(object)
}

required_columns <- c("meta_group")
missing_required <- setdiff(required_columns, colnames(object))
if (length(missing_required) > 0) {
  stop(sprintf(
    "df_cell_graph.rds lacks required columns: %s; available: %s",
    paste(missing_required, collapse = ", "),
    paste(colnames(object), collapse = ", ")
  ))
}

if ("cell_id" %in% colnames(object)) {
  cell_id <- as.character(object[["cell_id"]])
} else {
  object_rownames <- rownames(object)
  if (is.null(object_rownames) || identical(object_rownames, as.character(seq_len(nrow(object))))) {
    stop("df_cell_graph.rds has neither a cell_id column nor informative row names")
  }
  cell_id <- as.character(object_rownames)
}

keep_optional <- intersect(c("system", "celltype_new"), colnames(object))
result <- data.frame(
  cell_id = cell_id,
  graph_node = as.character(object[["meta_group"]]),
  stringsAsFactors = FALSE
)
for (column in keep_optional) {
  result[[column]] <- as.character(object[[column]])
}
rm(object)
invisible(gc())

valid <- !is.na(result$cell_id) & nzchar(result$cell_id) &
  !is.na(result$graph_node) & nzchar(result$graph_node)
result <- result[valid, , drop = FALSE]

if (anyDuplicated(result$cell_id) > 0) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    result_dt <- data.table::as.data.table(result)
    conflicts <- result_dt[, .(n_graph_nodes = data.table::uniqueN(graph_node)), by = cell_id][n_graph_nodes > 1]
    if (nrow(conflicts) > 0) {
      stop(sprintf("Found %d cell IDs mapped to multiple meta_group values", nrow(conflicts)))
    }
    result <- unique(result_dt, by = "cell_id")
  } else {
    duplicated_ids <- unique(result$cell_id[duplicated(result$cell_id)])
    for (identifier in head(duplicated_ids, 10000)) {
      if (length(unique(result$graph_node[result$cell_id == identifier])) > 1) {
        stop(sprintf("Cell ID %s maps to multiple meta_group values", identifier))
      }
    }
    result <- result[!duplicated(result$cell_id), , drop = FALSE]
  }
}

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary_path <- paste0(output_path, ".tmp")
message(sprintf("[rds] writing %d rows to %s", nrow(result), output_path))

if (grepl("\\.parquet$", output_path, ignore.case = TRUE)) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Writing Parquet requires the R package 'arrow'; use an OUTPUT.tsv.gz path instead")
  }
  arrow::write_parquet(result, temporary_path, compression = "zstd")
} else if (requireNamespace("data.table", quietly = TRUE)) {
  data.table::fwrite(result, temporary_path, sep = "\t", quote = FALSE, compress = "gzip")
} else {
  connection <- gzfile(temporary_path, open = "wt", compression = 6)
  write.table(result, file = connection, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
  close(connection)
}

if (!file.rename(temporary_path, output_path)) {
  stop(sprintf("Could not atomically rename %s to %s", temporary_path, output_path))
}
message(sprintf("[rds] success: %s", output_path))
