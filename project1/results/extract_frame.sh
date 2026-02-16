#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_directory> <output_directory> <frame_number>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
FRAME="$3"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory '$OUTPUT_DIR' does not exist. Creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

COUNT=0
for FILE in "$INPUT_DIR"/*.mp4; do
    [ -f "$FILE" ] || continue

    BASENAME=$(basename "$FILE")
    NAME="${BASENAME%.mp4}"

    echo "Extracting frame $FRAME from: $BASENAME"
    ffmpeg -i "$FILE" -vf "select=eq(n\,$FRAME)" -vframes 1 -y "$OUTPUT_DIR/${NAME}_frame${FRAME}.png"
    COUNT=$((COUNT + 1))
done

echo "Done. Extracted frame $FRAME from $COUNT file(s)."
