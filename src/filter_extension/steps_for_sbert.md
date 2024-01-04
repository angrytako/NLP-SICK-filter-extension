conda create -n sbert
conda activate sbert
conda install -c conda-forge sentence-transformers

# Optional; For this step you need to have cuda installed and check cuda version
install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 

conda intall datasets