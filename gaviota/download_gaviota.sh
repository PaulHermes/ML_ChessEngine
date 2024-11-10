#!/bin/bash

# Set base URL and folders
base_url="https://chess.cygnitec.com/tablebases/gaviota/"
subdirectories=("3" "4" "5")
download_folder="./gaviota_downloads"
extract_folder="./gaviota_tablebases"

# Create folders if they don't exist
mkdir -p "$download_folder"
mkdir -p "$extract_folder"

# Loop through each subdirectory
for subdir in "${subdirectories[@]}"; do
    url="${base_url}${subdir}/"
    
    # Get the list of .7z files or parts from the current subdirectory
    echo "Fetching list of files from $url..."
    if [ "$subdir" == "5" ]; then
        # Match parts in /5 (e.g., 5.7z.001, 5.7z.002)
        file_list=$(curl -s "$url" | grep -oP '(?<=href=")[^"]+\.7z\.\d+')
    else
        # Match regular .7z files in other directories
        file_list=$(curl -s "$url" | grep -oP '(?<=href=")[^"]+\.7z')
    fi

    # Check if file_list is empty
    if [ -z "$file_list" ]; then
        echo "No .7z files found in $url. Skipping this directory."
        continue
    fi

    # Download each file from the current subdirectory
    echo "Downloading Gaviota tablebase files from $url..."
    for file in $file_list; do
        echo "Downloading $file from $url..."
        curl -o "$download_folder/$file" "$url$file"
    done
done

# Concatenate and extract files from /5
if [ -n "$(ls $download_folder/5.7z.* 2>/dev/null)" ]; then
    echo "Concatenating parts for 5.7z..."
    cat "$download_folder/5.7z."* > "$download_folder/5.7z"

    # Extract the concatenated file directly into extract_folder
    echo "Extracting 5.7z..."
    "/c/Program Files/7-Zip/7z.exe" x "$download_folder/5.7z" -o"$extract_folder" -y -r
fi

# Extract remaining .7z files directly into the extract_folder
echo "Extracting other files..."
for file in "$download_folder"/*.7z; do
    # Skip the already extracted 5.7z
    if [ "$file" == "$download_folder/5.7z" ]; then
        continue
    fi

    echo "Extracting $file..."
    "/c/Program Files/7-Zip/7z.exe" x "$file" -o"$extract_folder" -y -r
done

# Clean up downloaded .7z files and parts
echo "Cleaning up..."
rm "$download_folder"/*.7z "$download_folder"/5.7z.*

echo "Done! All tablebase files are extracted to $extract_folder"
