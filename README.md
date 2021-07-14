# Scalable, Axiomatic Explanations of Deep Alzheimer's Diagnosis from Heterogeneous Data

[![Paper](https://img.shields.io/badge/arXiv-2107.05997-b31b1b)](https://arxiv.org/abs/2107.05997)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code to the paper "Scalable, Axiomatic Explanations of Deep Alzheimer's Diagnosis from Heterogeneous Data."
If you are using this code, please cite:

```
@inproceedings(Poelsterl2021-svehnn,
  title     = {{Scalable, Axiomatic Explanations of Deep Alzheimer's Diagnosis from Heterogeneous Data}},
  author    = {P{\"{o}}lsterl, Sebastian and Aigner, Christina and Wachinger, Christian},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.05997},
}
```

## Installation

Install Python 3.7 and the dependencies listed in `requirements.txt`.

## Usage

## Synthetic Data

To quantitatively evaluate SVEHNN on synthetically generated point clouds, execute:
```
python -m svehnn.syntehtic.run
```

The results will be stored in `outputs-synthetic-seed868662447/metrics.csv`.


## Alzheimer's Disease Data

In this experiment, we used T1 structural brain MRI and tabular data from the
[Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/).
Since we are not allowed to share our data, you would need to process the data yourself.

1. We used [FreeSurfer 5.3](https://www.freesurfer.net/) to segment all T1 structural brain MRI,
and obtained point clouds of the Hippocampus by running the
[Marching Cubes algorithm](https://vtk.org/doc/nightly/html/classvtkDiscreteMarchingCubes.html)
on the binary segmentation masks. Finally, we randomly selected 1024 points from each
mesh.
2. As tabular data, we used the following fields from `ADNIMERGE.CSV`:
  - AGE
  - PTEDUCAT
  - PTGENDER
  - APOE4
  - ABETA
  - TAU
  - PTAU
  - FDG
  - AV45
3. We imputed missing values with scikit-learn's
[SimpleImputer](https://scikit-learn.org/0.22/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer)
and appended a binary missingness indicator for each variable with missing values.
4. Once you processed the data, train `svehnn.adni.model.WidePointNetClassifier` using standard PyTorch.
5. To run SVEHNN on the trained model, you need to implement `BaseDataModule` in
`svehnn/adni/data.py` and then run:
```
python -m svehnn.adni.run
```
The estimated Shapley Values will be stored in `sv_svehnn.pkl`.
