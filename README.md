# Supplementary code for ''FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training''

## How to install

You just need to download repository and install the requirements:

```
pip install -r requirements.txt
```

## FRUGAL

The source code for **FRUGAL** is located in the `frugal` directory. The file `proj_optimizer_templates.py` contains a template class for three types of projection: [Galore-like (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507) SVD projection (GaloreOptimizer), RandK projection (CoordOptimizer), and BAdam-like [BAdam-like (Luo et al., 2024)](https://arxiv.org/abs/2404.02827) blockwise projection (BlockOptimizer). In the files `adamw.py`, `lion.py`, and `sgd.py`, both the original algorithms and **FRUGAL** versions are implemented with all types of projections, using these algorithms as state-full components.

**FRUGAL** features several hyperparameters:

- `proj_params_lr_scale`: A multiplier for the learning rate applied to projectable parameters. It is set to `1.0` in all main experiments.

- `update_gap`: The frequency of state-full subspace updates. It is set to `200` in all main experiments, consistent with [Galore (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507).

- `density`: The fraction of the total space in Linear layers that is updated with a state-full optimizer. Its default value is `0.25`.

- `inactive_update_rule`: Strategy for updating the state-free subspace. The options include 'no' for no update, 'sgd', and 'sign_sgd' for optimization using SGD and [signSGD (Bernstein et al., 2018)](https://arxiv.org/abs/1802.04434) respectively. Default value is `sign_sgd`.

- `inactive_lr_scale`: A multiplier for the learning rate on state-free parameters. It is set to `1.0` for pre-training and `0.1` for fine-tuning in main experiments.

Additionally, there are parameters specific to the types of projections:

- For `GaloreOptimizer`, there are parameters `proj_side` and `proj_type`. The `proj_side` parameter, derived from [Galore (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507), determines which matrix from the SVD is used for projection onto the low-rank subspace. The `proj_type` parameter allows for selecting among three projection matrices: `svd`, `random`, and `randperm` for SVD-like, random semi-orthogonal, and random permutation of the columns, respectively. Default value is `svd`.

- For `CoordOptimizer`, the type of projection can be chosen: `randk` for RandK projection on random coordinates within the Linear layer matrix, and `rows` and `columns` for projection onto entire random rows or columns. Its default value is `randk`.

- For `BlockOptimizer`, the order of selecting state-free active transformer blocks can be specified. Options include `random`, `descending`, `ascending`, and `mirror`, with `random` as default value.


## How to run

### Pre-training experiments

The scripts for running experiments on the [LLaMA-like model (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) pre-training  on the [C4 dataset (Raffel et al., 2020)](https://arxiv.org/abs/1910.10683) can be found in `scripts/benchmark_c4`. The main code for the experiments is located in `torchrun_main.py`. 

The optimization algorithm can be selected using the `optimizer` argument. Available options include `adamw`, `lion`, and `sgd`, along with their **FRUGAL** versions as state-full algorithms. You can choose these, for example, as `galore_adamw`, `coord_adamw`, and `block_adamw` (last also can be launched with `frugal`).

In addition to arguments specific to **FRUGAL**, you can also specify several other standard arguments such as `batch_size`, `warmup_steps`, `weight_decay`, `lr`, `scheduler`, `scheduler_cycle_length`, `num_training_steps`, among others. You can view the full list of arguments in `torchrun_main.py`.

One should also note the `dtype` and `amp` arguments. The `dtype` argument determines the `torch.dtype` in which the model and optimizer state are stored, while `amp` enables Automatic Mixed Precision training. In our main experiments, unlike in [Galore (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507), we used AMP training with dtype=fp32.

To collect gradients for reproducing Figure 2, make sure to enable the `collect_grads` flag.

Running baselines:

- For [Galore (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507), set `optimizer=galore_adamw` and specify the following: `reset_statistics=False`, `inactive_update_rule="no"`, `lr=0.01`, `proj_params_lr_scale=0.25`, and `density=0.25` (see Appendix A for details on `density`).

- For [BAdam (Luo et al., 2024)](https://arxiv.org/abs/2404.02827), set `optimizer=badam` and choose `block_order=descending`.

- For full-rank training specify `optimizer=adam`.

The code for pre-training experiments is based on the [Galore repository](https://github.com/jiaweizzhao/GaLore/). We are grateful to them for making their codebase available in the public domain.

### Fine-tuning experiments

Scripts for reproducing the experimental results on fine-tuning RoBERTa [Liu et al., 2019](https://arxiv.org/abs/1907.11692) on the GLUE benchmark [Wang et al., 2018](https://arxiv.org/abs/1804.07461) are located in the `scripts/glue` folder. In this folder, you can find scripts to run experiments with `rank=8` and `rank=0`. Note that, unlike the pre-training experiments, the `density` parameter takes very small values, so we have set it to allow specifying `density` through the `rank`. For details, see Section 5.2 and Appendix A.2.

The main code for fine-tuning is in `run_glue.py` and is an adaptation of the `run_glue.py` file from `transformers` library. The `transformers.Trainer` is used for training, so in addition to arguments for **FRUGAL**, you can specify standard arguments from the `TrainingArguments`, such as `gradient_accumulation_steps`, `fp16`, and others.

## Additional experiments

Notebook `principal_angles.ipynb` can be used to reproduce Figure 2. `galore_re-projection.ipynb` contains code for Appendix C experiments (Figure 3).
