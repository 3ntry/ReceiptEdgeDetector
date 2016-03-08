#!/bin/bash

# Make files
CMAKEFILES_DIR=./CMakeFiles
CMAKECACHE=/CMakeCache.txt
MAKEFILE=/Makefile

# Remove make files
if [[ -d "$CMAKEFILES_DIR" ]]; then
    rm -rf "$CMAKEFILES_DIR"
fi

if [[ -f "$CMAKECACHE" ]]; then
    rm -rf "$CMAKECACHE"
fi

if [[ -f "$MAKEFILE" ]]; then
    rm -rf "$MAKEFILE"
fi

# Compile
cmake .

make

IMAGEFILES_DIR=images/*

for f in $IMAGEFILES_DIR
do
    echo "processing $f"
    ./ReceiptScanner $f
done
