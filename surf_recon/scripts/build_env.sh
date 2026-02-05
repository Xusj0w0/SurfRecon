conda create -n gssurf python=3.10.12 -y
conda activate gssurf

# PyTorch
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    pytorch-cuda=11.8 mkl=2023.1.0 -c pytorch -c nvidia

# Depth Anything 3
cd utils/Depth-Anything-3
pip install xformers torch==2.5.1
pip install -e .
cd ../..

# PyTorch Lightning
pip install -r requirements/lightning.txt

# Dependencies
pip install -r requirements/common.txt --no-build-isolation

# Delaunay Triangulation
cd submodules/tetra_triangulation
conda install -y cmake==3.30.0 gmp cgal -c conda-forge
cmake .
make
pip install -e . --no-build-isolation
cd ../..

# PyTorch3D
cd submodules/pytorch3d
pip install -e . --no-build-isolation
cd ../..