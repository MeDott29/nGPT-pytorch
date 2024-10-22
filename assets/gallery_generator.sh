#!/bin/bash

# Define supported media file extensions
extensions=("jpg" "jpeg" "png" "gif")

# Start creating the README.md content
echo "# Image Gallery" > README.md
echo "This gallery displays the media files in the current directory." >> README.md
echo "" >> README.md

# Loop through each supported extension
for ext in "${extensions[@]}"
do
  # Find all media files with the current extension
  for file in *.$ext
  do
    # Check if the file exists to avoid iterating on unmatched patterns
    if [ -f "$file" ]; then
      # Replace spaces in filenames with underscores for proper Markdown linking
      safe_file=$(echo "$file" | sed 's/ /_/g')
      mv "$file" "$safe_file"
      echo "![${safe_file}](./${safe_file})" >> README.md
      echo "" >> README.md
    fi
  done
done

echo "Image gallery has been successfully created in README.md."
