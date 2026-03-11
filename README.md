# Supplemental Materials for Submission #5995

<p align="center">
  <img src="assets/demo.gif" width="640" />
</p>


## 1. Installation

```bash
bash surf_recon/scripts/build_env.sh
```

## 2. Data Preparation

1. Download and process datasets following [CityGS](https://github.com/Linketic/CityGaussian). The propocessed dataset folder structure is:
    ```
    ├── data_citygs
      ├── geometry_gt  # ground-truth pointcloud
        ├── SMBU
          ├── SMBU_ds.json
          ├── SMBU_ds.ply
          ├── transform.txt
        ├── LFLS
        ├── CUHK_UPPER
        ├── MC_Aerial
        ├── MC_Street
      ├── SMBU
          ├── images
          ├── sparse
            ├── 0
              ├── cameras.bin
              ├── images.bin
              ├── points3D.bin
      ├── LFLS
      ├── CUHK_UPPER_CAMPUS
      ├── mc_aerial
      ├── mc_street
   ```
2. Downsample images and modify sparse model accordingly. Take the Russian Building scene (SMBU) as an example:
    ```bash
    python surf_recon/tools/convert_dataset.py \
      --input data_citygs/SMBU \
      --output_dir data/GauU_Scene/SMBU \
      --rescale_width 1600
    ```
3. Predict depth prior using DepthAnything3. Take the Russian Building scene (SMBU) as an example:
    ```bash
    python surf_recon/tools/run_depth_anything_v3.py data/GauU_Scene
    ```
    The dataset folder structure for SMBU is expected to be:
    ```
    ├── SMBU
      ├── images
      ├── sparse
      ├── da3_depths
      ├── da3_inverse_depths
      da3_depths-scales.json
      da3_inverse_depths-scales.json
      val_images.txt
    ```

## 3. Run

Take the Russian Building scene (SMBU) as an example.

1. Train a coarse global model.
    ```bash
    python main.py fit --config surf_recon/configs/smbu/coarse.yaml
    ```
2. Partition the scene.
    ```bash
    python surf_recon/tools/partition_scene.py \
      -c surf_recon/configs/smbu/partition.yaml \
      -p smbu \
      -d data/GauU_Scene/SMBU
    ```
3. Train each block in parallel.
    ```bash
    python surf_recon/tools/train_partitions.py \
      -p smbu \
      -c surf_recon/configs/smbu/fine.yaml
    ```
4. Merge per-block models.
    ```bash
    python surf_recon/tools/merge_partitions.py -p smbu
    ```
    The merged checkpoint will be placed at `outputs/smbu/checkpoints/merged.ckpt`

## 4. Evaluation

Take the Russian Building scene (SMBU) as an example.

### 4.1. Geometric Accuracy

1. Extract surface from the merged model.
    ```bash
    python surf_recon/tools/extract_mesh.py \
      -c outputs/smbu/checkpoints/merged.ckpt \
      -d data/GauU_Scene/SMBU \
      -o outputs/smbu/mesh.ply
    ```
    The extracted surface will be placed at `outputs/smbu/mesh.ply`
2. Evaluate geometric accuracy
    ```bash
    python surf_recon/tools/eval_tnt/run.py \
      --dataset-dir data_citygs/geometry_gt/SMBU \
      --scene SMBU_ds \
      --transform-path data_citygs/geometry_gt/SMBU/transform.txt \
      --ply-path outputs/smbu/mesh.ply \
      --out-dir outputs/smbu/evaluations/geometry
    ```

### 4.2. Rendering Quality

1. Render images
    ```bash
    python surf_recon/tools/render.py \
      -c outputs/smbu/checkpoints/merged.ckpt \
      -d data/GauU_Scene/SMBU \
      -o outputs/smbu/evaluations/render
    ```
2. Evaluate rendering quality
    ```bash
    python surf_recon/tools/eval.py \
      --render outputs/smbu/evaluations/render/images/render
    ```