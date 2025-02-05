This work is the implementation of a conference paper [Deep Learning for Cross-Domain Building Change Detection from Multi-Source Very High-Resolution Satellite Imagery](https://doi.org/10.1109/IGARSS53475.2024.10641261). Except for domain adaptation and some data loader sections, this repository is forked from the [codebase](https://github.com/wgcban/SemiCD) used for the implementation of [Bandara and Patel (2022)](https://arxiv.org/abs/2204.08454). On top of their work, we introduced an optimal transport [Courty et al.(2016)](https://doi.org/10.1109/TPAMI.2016.2615921) domain adaptation approach for cross-domain change detection workflow. To implement domain adaptation we used an implementation from a dedicated package [POT: Python Optimal Transport](https://pythonot.github.io/).

# usage
train

``` python train.py --config /path2configfile ```

inference
```python inference.py --config /path2configfile ```

feature space plot
```python generate_features.py --save_root /root2save --data_root /root2data```
```python feature_space_plot.py --save_root /root2save --data_root /root2data```

# citation 

The work can be cited as:

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
