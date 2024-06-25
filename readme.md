# EIP for MoCo
MRI motion correction for generalizable scenarios.

This is the official PyTorch implementation of our manuscript:

> [**Generalizable MRI Motion Correction via Compressed Sensing Equivariant Imaging Prior**](xxx)

## Getting started

### Todo
Update the simulaton code and data.

###  1. Clone the repository
```bash
git clone https://github.com/wangzhiwen-scu/EIP4MoCo.git
cd eip
```

### 2. Install dependencies

Here's a summary of the key dependencies.
- python 3.7
- pytorch 1.7.1

We recommend using [conda](https://docs.conda.io/en/latest/) to install all of the dependencies.

```bash
conda env create -f environment.yaml
```
To activate the environment, run:

```bash
conda activate eip
```

### 3. Pre-trained Model and Testing Dataset
All data and models can be downloaded in [Google-drive](https://drive.google.com/file/d/1tyX2oOUIyLwx3HImF0lDsEUjNqMIennG/view?usp=sharing).

It is a `eip_demo_testing_files` zip file (~414M) which contain a demo dataset and a group of demo parameters. Unzip it we can get `exp1` folder and `ixi_t1_sudden_moderate` folder. `exp1` contains network parameters of compared methods (MARC, Med-cycleGAN, BSA and supvised method.) and our methods. `ixi_t1_sudden_moderate` contains IXI dataset with T1 modality sudden motion and GT data.

### 4. File Organization
we can see two datasets folders:  `IXI_T1_motion` and `MA-ART`, where `IXI_T1_motion` is simulation motion datasets and `MA-ART` is real motion datasets.

They are organized in:

```
├── data
│   ├── bezier_curve.py
│   ├── build_h5py_data.py
│   ├── build_h5py_data_threeslices.py
│   ├── cplx_data
│   ├── dataset.py
│   ├── datasets
        ├── IXI_T1_motion
        └── MA-ART
│   ├── generate_mask.py
│   ├── __init__.py
│   ├── masks
│   ├── __pycache__
└── └── toolbox.py
```

For real motion: download MA-ART [Dataset Link](https://openneuro.org/datasets/ds004173/versions/1.0.2). Then place the datasets in ```./data/datasets/MR-ART ```.

For simulation motion: download `ixi_t1_sudden_moderate` file above mentioned; then place the `ixi_t1_sudden_moderate` in:

```
IXI_T1_motion
└── ixi_t1_sudden_moderate
```

create a `model_zoo` folder: `mkdir model_zoo`, place the `exp1` (parameter files) in:
```
├── model_zoo
└──└── exp1
```
### 5. Training

Please see [training/demo.sh](training/demo_new.sh) for an example of how to train EIP.

### 6. Testing

```
bash ./testing/testing.sh
```

## Acknowledgement

Part of the data simulation are adapted from **EI**. 
Part of the learning network are adapted from **MRI-Motion-Artifact-Simulation-Tool** and **RetroMoCoBox**.
Part of the plug-and-play motion estimation code are adapted from **NAMER**. 

<!-- Part of the reconstruction network structures are adapted from **MD-Recon-Net**. -->
 
+ MA-ART (Real motion datasets): Paper-[https://www.nature.com/articles/s41597-022-01694-8#code-availability](https://www.nature.com/articles/s41597-022-01694-8); Code-[https://openneuro.org/datasets/ds004173/versions/1.0.2](https://openneuro.org/datasets/ds004173/versions/1.0.2).
+ MRI-Motion-Artifact-Simulation-Tool (Rigid motion simulation): [https://github.com/Yonsei-MILab/MRI-Motion-Artifact-Simulation-Tool](https://github.com/Yonsei-MILab/MRI-Motion-Artifact-Simulation-Tool).
+ RetroMoCoBox: [https://github.com/dgallichan/retroMoCoBox](https://github.com/dgallichan/retroMoCoBox) for phase simulation and motion simulation.
+ EI: [https://github.com/edongdongchen/EI](https://github.com/edongdongchen/EI).
+ NAMER: [https://github.com/mwhaskell/namer_MRI](https://github.com/mwhaskell/namer_MRI) for multicoil motion parameters estimation.

Thanks a lot for their great works!

## contact
If you have any questions, please feel free to contact Wang Zhiwen {wangzhiwen_scu@163.com}.

 <!-- ## Citation

If you find this project useful, please consider citing:

```bibtex
@article{wang2024promoting,
  title={Promoting fast MR imaging pipeline by full-stack AI},
  author={Wang, Zhiwen and Li, Bowen and Yu, Hui and Zhang, Zhongzhou and Ran, Maosong and Xia, Wenjun and Yang, Ziyuan and Lu, Jingfeng and Chen, Hu and Zhou, Jiliu and others},
  journal={Iscience},
  volume={27},
  number={1},
  year={2024},
  publisher={Elsevier}
}
``` -->
