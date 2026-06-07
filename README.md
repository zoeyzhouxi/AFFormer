# AFFormer

Adaptive Feature Fusion Transformer for V2X Cooperative Perception under Channel Impairments

Paper Link: https://arxiv.org/abs/2605.01888

## Abstract

Accurate 3D object detection is essential for ensuring the safety of autonomous vehicles. 
Cooperative perception, which leverages vehicle-to-everything (V2X) communication to share perceptual data, enhances detection but is vulnerable to channel impairments, such as noise, fading, and interference. 
To strengthen the reliability of intelligent transportation systems, this work improves the robustness of V2X cooperative perception under communication conditions that reflect common channel impairments.
This paper proposes an Adaptive Feature Fusion Transformer (AFFormer), a Transformer-based framework that mitigates the adverse effects of corrupted features by modeling temporal, inter-agent, and spatial correlations. 
AFFormer introduces three key modules: Multi-Agent and Temporal Aggregation for context-aware fusion across agents and over time, Dual Spatial Attention for efficient modeling of spatial dependencies, and Uncertainty-Guided Fusion for entropy-driven refinement of fused features. 
A teacher–student knowledge distillation strategy further enhances robustness by aligning fused features with reliable early-collaboration supervision.

## Installation
You can also refer to [CoAlign](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) and [OpenCOOD](https://opencood.readthedocs.io/en/latest/md_files/installation.html) Installation Guide to learn how to install and run this repo.

### 1. Create a conda environment

All operations should be done on machines with GPUs. We expect a CUDA version displayed in the top right corner of the ```nvidia-smi``` to be higher than ```11.8```

```
conda create -n torch39 python=3.9
conda activate torch39

conda config --add channels conda-forge
conda config --set channel_priority flexible

# Install a matching stack for cu118
conda install -y pytorch=2.5.1 torchvision=0.20.1 

conda install git-lfs boost cuda cuda-nvcc mkl=2023.1.0 cmake -c conda-forge -c nvidia/label/cuda-11.8

conda install -y "opencv=4.10.0=py39h0a8ef67_0"
conda install -y "setuptools=75.1.0=py39h06a4308_0"

conda install -c nvidia cuda-toolkit=11.8
```

### 2. Install spconv 2.x 
Install the package that matches your cuda toolkit version:
```
pip install spconv-cu118
```

### 3. Install some other packages
```
pip install -r requirements.txt
```

### 4. Clone AFFormer
```
git clone https://github.com/zoeyzhouxi/AFFormer.git
cd AFFormer
export PYTHONPATH=${AFFormer_Folder}:$PYTHONPATH
```

### 5. Bbx IOU cuda version compile
Install bbx nms calculation cuda version
```
python afformer/utils/setup.py build_ext --inplace
```
This step are the same as [Step 4 in OpenCOOD Installation Guide](https://opencood.readthedocs.io/en/latest/md_files/installation.html#bbx-iou-cuda-version-compile)


## Data Preparation
mkdir a `dataset` folder under AFFormer. Put your V2XSet and DAIR-V2X data in this folder. 

```
AFFormer/dataset

. 
├── dairv2x 
│   ├── cooperative
│   ├── infrastructure-side
│   ├── vehicle-side
│   ├── train.json
│   └── val.json
└── v2xset
    ├── test
    ├── train
    └── validate
```

Note that we use complemented annotation for DAIR-V2X in `dairv2x`，please check [CoAlign](https://github.com/yifanlu0227/CoAlign) for more details.


## Training

### 1. Train the teacher model

```
python afformer/tools/train.py --hypes_yaml afformer/hypes_yaml/v2xset/pointpillar_early.yaml --fusion_method early [--model_dir ${CHECKPOINT_FOLDER}]
```
then the teacher model will be saved in `{teacher_model_path}`.

### 2. Train the student model 
First Set parameter of `kd_flag->teacher_path` to `{teacher_model_path}` in `pointpillar_afformer.yaml`. set `model->args->fading` to `false` to train on perfect setting. Run the following command to train the student model.
```
python afformer/tools/train_w_kd.py --hypes_yaml afformer/hypes_yaml/v2xset/pointpillar_afformer.yaml
```

After the model on perfect setting converged, keep only the last checkpoint and rename it as `net_epoch0.pth`, modify the config yaml in your trained model directory to set `model->args->backbone_fix` to `true` (to fix the parameters of the encoder on each single-agent) and `model->args->fading` to `true` to train on fading setting.

```
python afformer/tools/train_w_kd.py --hypes_yaml afformer/hypes_yaml/v2xset/pointpillar_afformer.yaml --model_dir ${CHECKPOINT_FOLDER}
```

## Checkpoints

- [AFFormer DAIR-V2X](https://drive.google.com/drive/folders/1CFDw_2zqXmnbfJxhvP3FDLokzhSydcjl?usp=sharing)
- [AFFormer V2XSet](https://drive.google.com/drive/folders/1XJTKMv3-UKWxbzCd6ocuRpbOiqek1SH_?usp=sharing)

Download them and save them to `afformer/logs/`.

## Citation

```
@misc{zhou2026afformer,
      title={AFFormer: Adaptive Feature Fusion Transformer for V2X Cooperative Perception under Channel Impairments}, 
      author={Xi Zhou and Tao Huang and Qing-Long Han and Rana Abbas and Mostafa Rahimi Azghadi},
      year={2026},
      eprint={2605.01888},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.01888}, 
}
```

## Acknowlege

This project is impossible without the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoAlign](https://github.com/yifanlu0227/CoAlign)!

Many thanks to [@DerrickXuNu](https://github.com/DerrickXuNu) and [@yifanlu0227](https://github.com/yifanlu0227) for the great code framework.