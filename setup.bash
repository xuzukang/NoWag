

# conda create -n NoWAG -y
# conda activate NoWAG

# conda install pip -y
~/miniconda3/envs/NoWAG/bin/pip3 install torch torchvision torchaudio
conda install datasets matplotlib numpy pandas scikit-learn seaborn tqdm transformers lightning -c conda-forge -y