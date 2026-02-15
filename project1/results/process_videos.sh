#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

COUNT=0
for FILE in "$INPUT_DIR"/*.mp4; do
    [ -f "$FILE" ] || continue

    BASENAME=$(basename "$FILE")
    NAME="${BASENAME%.mp4}"
    ESCAPED=$(echo "$NAME" | sed "s/&/\\\\&/g; s/(/\\\\(/g; s/)/\\\\)/g; s/=/\\\\=/g; s/:/\\\\:/g")

    echo "Processing: $BASENAME"
    ffmpeg -i "$FILE" -vf "scale=iw*4:ih*4, drawtext=text='\#%{eif\:n\:d}':x=10:y=h-th-10:fontsize=24:fontcolor=white:borderw=2:bordercolor=black, drawtext=text='${ESCAPED}':x=10:y=10:fontsize=24:fontcolor=white:borderw=2:bordercolor=black" -y "$OUTPUT_DIR/$BASENAME"
    COUNT=$((COUNT + 1))
done

echo "Done. Processed $COUNT file(s)."
