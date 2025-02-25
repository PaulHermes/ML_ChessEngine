#!/bin/bash

# Check if 7-Zip is installed (Windows default path)
if [ ! -f "/c/Program Files/7-Zip/7z.exe" ]; then
    echo "7-Zip not found. Installing..."
    installer_url="https://www.7-zip.org/a/7z2301-x64.exe"
    installer_name="7z-installer.exe"

    # Download installer
    echo "Downloading 7-Zip installer..."
    if ! curl -L -o "$installer_name" "$installer_url"; then
        echo "Failed to download 7-Zip installer."
        exit 1
    fi

    # Run installer silently (requires admin rights)
    echo "Installing 7-Zip (this may require admin privileges)..."
    if ! "$installer_name" /S ; then
        echo "Failed to install 7-Zip. Please run as administrator or install manually."
        rm -f "$installer_name"
        exit 1
    fi

    # Cleanup installer
    rm -f "$installer_name"

    # Verify installation
    if [ ! -f "/c/Program Files/7-Zip/7z.exe" ]; then
        echo "7-Zip installation failed. Install manually from https://www.7-zip.org/"
        exit 1
    fi
fi

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
    echo "Concatenating parts for 5.7z in numerical order..."
    find "$download_folder" -name "5.7z.*" -print0 | sort -zt. -n -k3 | xargs -0 cat > "$download_folder/5.7z"

    # Extract files directly to root folder
    echo "Extracting 5.7z..."
    "/c/Program Files/7-Zip/7z.exe" e "$download_folder/5.7z" -o"$extract_folder" -y
    if [ $? -ne 0 ]; then
        echo "Error extracting 5.7z. File may be corrupted."
        exit 1
    fi
fi

# Extract remaining .7z files
echo "Extracting other files..."
for file in "$download_folder"/*.7z; do
    [ "$file" = "$download_folder/5.7z" ] && continue
    echo "Extracting $file..."
    "/c/Program Files/7-Zip/7z.exe" e "$file" -o"$extract_folder" -y
done

# Cleanup
rm -f "$download_folder"/*.7z*

echo "Done! All files extracted to $extract_folder"