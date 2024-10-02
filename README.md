# Installation

## Install GaLore optimizer

```bash
git clone git@github.com:fzmushko/LoRA.git
cd GaLore
pip install -e .
```

## Install experiment dependencies

```bash
pip install -r requirements.txt
```

## Launching the experiment

Bash scripts can be found in `scripts/benchmark_c4`. Main scipt is `llama_60m.sh`. Default setting requires 1 A100 80gb, but you can decrease batch_size to fit on the device with less memory.

## Algorithms code

Main code is located in the `galore_torch`. 

* `galore_reprojection.py` contains stuff for updating projection matrix and projecting gradient.
* `adamw.py` - IT IS NOT ORIGINAL ADAM.
Contains logic for galore/out method optimization step.
* `projected_adam.py` - "kind of" our algorithm. Reprojects first moment matrix M, but stores V in full space.
* `rot_adam.py` - "sanity check" reimplementation of GaLore for full rank runs

Other files are kept without modification from https://github.com/jiaweizzhao/GaLore.

## Arguments

There are multiple available optimizers.

* `adam` - normal AdamW from the torch library
* `galore_adamw` - both original GaLore and our method
    * `first_strategy`="no", second_strategy="no" - original GaLore
    * `first_strategy`="align_correction", second_strategy="reset" - our method
* `proj_adam` - kind of out method, but with full rank V
    * `first_strategy`="align_correction" - M in lower space
    * `first_strategy`="align_correction" - M in full space, but gradient is still projected to the first $r$ vectors of left singular basis. Difference with the previous choice is that parameters outside of the current projection image are still updated with momentum from previous iteration. WARNING: you should not enable compute_proj_metrics with this argument.
* `rot_adam` - "sanity check" reimplementation of GaLore for full rank runs

Other important parameters:
* `update_proj_gap` - gap between computing new projection matrix
* `galore_scale` - parameter from the original implementation. Changes learning rate to lr*galore_scale for parameters, that are updated with GaLore algorithm (note, that, biases, embedding/logit layers, etc. are updated with normal Adam). galore_scale 1.0 doesn't change lr.
* `rank` - rank of lower space. Usual choice for 60m llama is 128 (1/4 of full rank)
* `random_projection` - enables random projection. Otherwise uses svd projection.
* `amp` - enables Mixed Precision training.
