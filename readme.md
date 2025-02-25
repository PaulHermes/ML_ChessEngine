## Prerequisites

- **Miniconda:**  
  Download and install Miniconda for Windows from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).

## Installation

1. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yml

2. **Shell Initialization:**
    ```bash
    conda init bash
   
3. **Activate the environment:**
   ```bash
   conda activate tf

4. **Adjust parameters in parameters.py as needed. Currently it is set to Fine-Tuning(See training_regimen.md)**

5. **Downloading Gaviota Tablebases:**
    There are two methods to obtain the Gaviota tablebases:

   - **Automated Download (via Script):**  
     Ensure that 7Zip is installed in its default path, then execute:
     ```bash
     ./download_gaviota.sh
   - **Manual Installation:**  
     Download the tablebases from [here](https://chess.cygnitec.com/tablebases/gaviota/) and place them in the `gaviota_tablebases` directory.

## Usage

- **Generate Training Data:**

  ```bash
  python src/self_play.py
  
- **Train the Engine:**

  ```bash
  python src/training_pipeline.py
  
- **Play against the Engine:**

  ```bash
  python src/game_interface.py

## Additional Modules

If necessary, simply pip install additional Python modules. In particular, the following packages have caused issues:

- `chess`:  
    ```bash 
  pip install chess
- `tensorflow`  
    ```bash
    pip install tensorflow
