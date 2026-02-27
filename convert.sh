#!/bin/bash

# Usage: ./process_images.sh input.txt

INPUT_FILE="$1"

if [[ -z "$INPUT_FILE" ]]; then
  echo "Usage: $0 input.txt"
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

while IFS= read -r file; do
  # Skip empty lines
  [[ -z "$file" ]] && continue

  # If file does not exist, skip
  if [[ ! -f "$file" ]]; then
    echo "Skipping (not found): $file"
    continue
  fi

  ext="${file##*.}"
  base="${file%.*}"

  shopt -s nocasematch

  if [[ "$ext" == "jpg" || "$ext" == "jpeg" ]]; then
    png_file="${base}.png"
    echo "Converting: $file -> $png_file"
    convert "$file" "$png_file" && rm -f "$file"

  elif [[ "$ext" == "png" ]]; then
    echo "Deleting PNG: $file"
    rm -f "$file"

  else
    echo "Skipping (not jpg/png): $file"
  fi

done <"$INPUT_FILE"
