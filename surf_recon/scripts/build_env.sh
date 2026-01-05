conda create -n surf_recon python=3.10.12 -y

conda activate surf_recon
pip install -r requirements/pytorch.txt
pip install -r requirements/lightning.txt
pip install -r requirements/common.txt --no-build-isolation
