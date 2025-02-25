install miniconda: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

conda env create -f environment.yml

conda init bash (if you use bash)

conda activate tf

modify parameters to your liking, currently it is set to Fine-Tuning in training_regimen.md

to download gaviota tablebases:
    ./download_gaviota.sh

to generate training data:
    python src/self_play.py

to train:
    python src/training_pipeline.py

to play against the engine:
    python src/game_interface.py

pip install modules 