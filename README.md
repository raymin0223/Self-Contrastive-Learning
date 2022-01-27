# Self-Contrastive-Learning

This repository is the official PyTorch implementation of **"Self-Contrastive Learning: An Efficient Supervised Contrastive Framework with Single-view and Sub-network"**.

## Requirements

- This codebase is written for `python3` (used `python 3.7.6` while implementing).
- We use Pytorch version of `1.8.1` and `10.2`, `11.2` CUDA version.
- To install necessary python packages,  
    ```
    pip install -r requirements.txt
    ```

## How to Run Codes?

### Command

```
bash scripts/resnet_represent_selfcon.sh
```
    
- In `./scripts` directory, we made one example for SelfCon-S loss on ResNet-18 experiment.
- To train the other algorithms or architectures, change the appropriate argument referring to `parse_option()` in the main python files. 
- For example, if you want to train with the multi-viewed batch, change the `--multiview` argument to `True`.


## Reference Github

We refer to the following works:

- https://github.com/HobbitLong/SupContrast
- https://github.com/HobbitLong/RepDistiller
- https://github.com/facebookresearch/moco
- https://github.com/jiamings/ml-cpc
- https://github.com/UMBCvision/CompRess
