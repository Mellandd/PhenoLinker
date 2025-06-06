#!/bin/bash

# URLs of the files to download
PHENOTYPE_GENE_URL="https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2022-12-15/phenotype_to_genes.txt"

# Names to save the files as
PHENOTYPES_TO_GENE="phenotypes_to_genes.tsv"

# Download and rename the files
wget -O "$PHENOTYPES_TO_GENE" "$PHENOTYPE_GENE_URL"

sed -i '' '1s/.*/HPO-id\tHPO label\tentrez-gene-id\tentrez-gene-symbol\tAdditional Info from G-D source\tG-D source\tdisease-ID for link/' "$PHENOTYPES_TO_GENE"
echo "Files downloaded:"
echo "2. $PHENOTYPES_TO_GENE"