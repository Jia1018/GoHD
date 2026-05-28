# BFM (Basel Face Model)

The 3DMM coefficient detector relies on Basel Face Model 2009. These files are
**not redistributed** because they are released under a research-only license.

Place the following files in this folder before running inference:

```
BFM/
├── 01_MorphableModel.mat       # original BFM09
├── BFM_exp_idx.mat             # vertex indices for expression basis
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin                 # expression basis from FaceWarehouse / Guo et al.
├── facemodel_info.mat
├── select_vertex_id.mat
├── similarity_Lm3D_all.mat
└── std_exp.txt                 # already shipped with this repo
```

## How to obtain them

1. **`01_MorphableModel.mat`** — request access from the
   [Basel Face Model](https://faces.dmi.unibas.ch/bfm/) website.
2. **`Exp_Pca.bin`** — provided with the
   [FaceWarehouse / 3DMM-Fitting-Pytorch](https://github.com/Juyong/3DFace) project.
3. **The remaining `.mat` files** are the index / mask tables used by
   [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch). They are
   also mirrored in the original GoHD release at
   <https://drive.google.com/drive/folders/1S2RxB8pUsO-lM4iRi6rO7EPdDpM85h0k>
   under `BFM/`, plus an additional copy under
   `Deep3DFaceRecon_pytorch/BFM/`.

After downloading, you should have a `BFM_model_front.mat` of roughly 280 MB.
