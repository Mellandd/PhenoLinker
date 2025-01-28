#!/bin/bash

# URLs of the files to download
FILE1_URL="https://example.com/path/to/firstfile"
PHENOTYPE_GENE_URL="https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2022-12-15/phenotype_to_genes.txt"

# Names to save the files as
FILE1_NAME="renamed_first_file"
PHENOTYPES_TO_GENE="phenotypes_to_genes.tsv"

# Download and rename the files
wget -O "$FILE1_NAME" "$FILE1_URL"
wget -O "$PHENOTYPES_TO_GENE" "$PHENOTYPE_GENE_URL"

echo "Files downloaded:"
echo "1. $FILE1_NAME"
echo "2. $PHENOTYPES_TO_GENE"