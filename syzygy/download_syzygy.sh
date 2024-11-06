#!/bin/bash

# Base URLs
dtz_url="https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/"
wdl_url="https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/"

# Folders to save downloads
download_folder_dtz="./"
download_folder_wdl="./"

# Create directories if they don't exist
mkdir -p "$download_folder_dtz"
mkdir -p "$download_folder_wdl"

# Function to download all files from a URL
download_files() {
    local url=$1
    local folder=$2

    # Get the HTML page with the list of files
    file_list=$(curl -s "$url" | grep -oP '(?<=href=")[^"]*\.rtb[zw]')

    # Iterate through each file and download it
    for file in $file_list; do
        full_url="${url}${file}"
        local_path="${folder}/${file}"

        # Download each file
        echo "Downloading $full_url to $local_path"
        curl -o "$local_path" "$full_url"
    done
}

# Download DTZ files
download_files "$dtz_url" "$download_folder_dtz"

# Download WDL files
download_files "$wdl_url" "$download_folder_wdl"
