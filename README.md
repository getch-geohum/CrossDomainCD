This repository is an implementation of [Deep Learning for Cross-Domain Building Change Detection from Multi-Source Very High-Resolution Satellite Imagery](https://doi.org/10.1109/IGARSS53475.2024.10641261). The repository has used most of impk,ementations from [Bandara and Patel (2022)](https://github.com/wgcban/SemiCD implementatation for a paper [evisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images](https://arxiv.org/abs/2204.08454). on top of their implementation, this repository introduces concept of domain adaptation [Courty et al.](https://doi.org/10.1109/TPAMI.2016.2615921) for cross domain chnage detection. The optimal transport domain adaptation is implemented using a dedicated python package [POT: Python Optimal Transport](https://pythonot.github.io/)

# usage
training 
`` ```
inference
``` ```
feature space plot 
``` ```
# citation

If you use this repo, please cite:
```
@inproceedings{gella2024deep,
  title={Deep Learning for Cross-Domain Building Change Detection from Multi-Source Very High-Resolution Satellite Imagery},
  author={Gella, Getachew Workineh and Lang, Stefan},
  booktitle={IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
  pages={421--426},
  year={2024},
  organization={IEEE}
}
```
