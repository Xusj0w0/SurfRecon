conda create -n gssurf python=3.10.12 -y

conda activate gssurf
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    pytorch-cuda=11.8 mkl=2023.1.0 -c pytorch -c nvidia
pip install -r requirements/lightning.txt
pip install -r requirements/common.txt --no-build-isolation

# nerf tetra
cd submodules/tetra_triangulation
conda install -y cmake==3.30.0 conda-forge::gmp conda-forge::cgal
cmake .
make
pip install -e . --no-build-isolation
cd ../..
