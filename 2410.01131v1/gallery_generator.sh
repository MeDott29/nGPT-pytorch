#!/bin/bash

# Define supported media file extensions
extensions=("jpg" "jpeg" "png" "gif")

# Create the assets folder if it doesn't already exist
mkdir -p assets

# Start creating the README.md content
echo "# Image Gallery" > README.md
echo "This gallery displays the media files located in the 'assets' folder." >> README.md
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
      mv "$file" "assets/$safe_file"
      echo "![${safe_file}](./assets/${safe_file})" >> README.md
      echo "" >> README.md
    fi
  done
done

echo "Image gallery has been successfully created in README.md."
