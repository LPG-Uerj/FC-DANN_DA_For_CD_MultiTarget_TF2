#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <extension>"
    exit 1
fi

DIRECTORY=$1
EXTENSION=$2

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Find and delete files with the specified extension
find "$DIRECTORY" -type f -name "$EXTENSION" -exec rm -f {} \;

echo "All files with extension .$EXTENSION in $DIRECTORY and its subfolders have been deleted."