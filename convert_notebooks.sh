#!/bin/bash

# Script to convert Jupyter notebooks to Python scripts and archive originals
# Usage: ./convert_notebooks.sh

# Set the base directory
BASE_DIR="$(pwd)"
ARCHIVE_DIR="${BASE_DIR}/archive"

# Ensure archive directory exists
mkdir -p "${ARCHIVE_DIR}"

# Find all .ipynb files
find "${BASE_DIR}" -name "*.ipynb" -type f | while read -r notebook_path; do
    # Skip files that are already in the archive directory
    if [[ "${notebook_path}" == "${ARCHIVE_DIR}"* ]]; then
        echo "Skipping file in archive: ${notebook_path}"
        continue
    fi

    echo "Processing: ${notebook_path}"
    
    # Get the relative path from the base directory
    rel_path="${notebook_path#${BASE_DIR}/}"
    
    # Create the directory structure in the archive
    archive_dir="$(dirname "${ARCHIVE_DIR}/${rel_path}")"
    mkdir -p "${archive_dir}"
    
    # Convert the notebook to Python
    echo "  Converting to Python..."
    jupyter nbconvert --to python "${notebook_path}"
    
    # Move the original notebook to the archive
    echo "  Moving original to archive..."
    mv "${notebook_path}" "${ARCHIVE_DIR}/${rel_path}"
    
    echo "  Done!"
done

echo "All notebooks processed successfully!"
echo "Converted notebooks are in their original locations as .py files"
echo "Original notebooks are archived in ${ARCHIVE_DIR}"

