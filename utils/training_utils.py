import os 

import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

import transformers
import wandb

from frugal import AdamW, GaloreAdamW, CoordAdamW, BlockAdamW
from frugal import SGD, GaloreSGD, CoordSGD, BlockSGD
from frugal import Lion, GaloreLion, CoordLion, BlockLion

from badam import BlockOptimizer as BAdamBlockOptimizer

def get_scheduler(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "constant":
        return transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor = tensor * random_pruning_mask
    return tensor


@torch.no_grad()
def magnitude_pruning(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor = tensor * mask.to(dtype=tensor.dtype)
    return tensor


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps < restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)

def get_num_layers(module):
    for name, children in module.named_children():
        if isinstance(children, torch.nn.ModuleList):
            return len(children)
        else:
            res = get_num_layers(children)
            if isinstance(res, int):
                return res
            
def get_optimizer(param_groups, args, model=None):
    assert args.optimizer.lower() != "badam" or model is not None, "BAdam requires model for the initialization."
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)#, foreach=False, fused=False)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaloreAdamW(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer.lower() == "coord_adamw":
        optimizer = CoordAdamW(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer.lower() == "block_adamw":
        optimizer = BlockAdamW(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer.lower() == "lion":
        optimizer = Lion(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_lion":
        optimizer = GaloreLion(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "coord_lion":
        optimizer = CoordLion(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "block_lion":
        optimizer = BlockLion(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(param_groups, lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif args.optimizer.lower() == "galore_sgd":
        optimizer = GaloreSGD(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif args.optimizer.lower() == "coord_sgd":
        optimizer = CoordSGD(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            coord_choice=args.coord_choice,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif args.optimizer.lower() == "block_sgd":
        optimizer = BlockSGD(
            param_groups, 
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif args.optimizer.lower() == "badam":
        original_optimizer = torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, foreach=False, fused=False)
        num_layers = len(model.model.layers)
        block_size = int(args.density * num_layers)
        if num_layers % block_size:
            raise ValueError("Incorrect density - can't split layers in equal parts.")
        block_prefix_list = []
        for block_start in range(0, num_layers, block_size):
            cur_block_prefix_list = []
            for idx in range(block_start, block_start + block_size):
                cur_block_prefix_list.append(f"model.layers.{idx}.self_attn.")
                cur_block_prefix_list.append(f"model.layers.{idx}.mlp")
            block_prefix_list.append(cur_block_prefix_list)
        optimizer = BAdamBlockOptimizer(
            base_optimizer=original_optimizer,
            named_parameters_list=list(model.named_parameters()), 
            block_prefix_list=block_prefix_list,
            active_modules=[name for name, _ in model.named_parameters() if "embed" in name or "norm" in name or "head" in name],
            switch_block_every=args.update_gap,
            switch_mode=args.block_order,
            verbose=2,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer

def get_density(model, optimizer_args):
    assert optimizer_args.rank is not None
    for p in model.classifier.parameters():
        optimizer_args.density = optimizer_args.rank / min(p.size())
        break

def is_distributed_environment():
    return 'LOCAL_RANK' in os.environ

def get_dist_rank():
    return dist.get_rank()

class TrainerWithWandbRestart(transformers.Trainer):
    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
            if 'wandb' in self.args.report_to:
                with open (f'{self.args.output_dir}/run_id.txt', 'w') as f:
                    f.write(f'{wandb.run.id}')
                    