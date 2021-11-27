eval "$(conda shell.bash hook)"
conda activate autoencoder &&
python FlowAutoencoderFC.py -g 3 -b 40 -e 200 --z_dim 512 &&
python FlowAutoencoderFC.py -g 3 -b 40 -e 200 --z_dim 128