import os
import sys
from datetime import datetime

from loguru import logger

import torch

def check_args_torchrun_main(args):
    if args.optimizer.lower() == "frugal":
        args.optimizer = "block_adamw"
    if (args.proj_embeds or args.proj_logits or args.proj_norms) and not "block" in args.optimizer:
        raise ValueError("proj embeds/logits/norms are implemented only with block optimizer")

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")
    
    if args.beta2 is None:
        if "lion" in args.optimizer:
            args.beta2 = 0.99
        elif "adam" in args.optimizer:
            args.beta2 = 0.999

    if not len(args.wandb_tags):
        get_pretraining_name_and_tags(args)

    if args.save_dir is None:
        args.save_dir = args.wandb_name if args.save_dir_prefix is None else os.path.join(args.save_dir_prefix, args.wandb_name)

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py")

    elif args.dtype in ["bf16", "bfloat16"]:
        args.dtype = getattr(torch, "bfloat16")
    elif args.dtype in ["fp32", "float32"]:
        args.dtype = getattr(torch, "float32")
    else:
        raise ValueError

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    return args

def check_args_trainer_finetuning(training_args, optimizer_args):
    if optimizer_args.optimizer.lower() == "frugal":
        optimizer_args.optimizer = "coord_adamw"
    if optimizer_args.beta2 is None:
        if "lion" in optimizer_args.optimizer:
            optimizer_args.beta2 = 0.99
        elif "adam" in optimizer_args.optimizer:
            optimizer_args.beta2 = 0.999

    get_finetuning_name_and_tags(training_args, optimizer_args)

    training_args.output_dir += training_args.run_name

    optimizer_args.lr = training_args.learning_rate
    optimizer_args.weight_decay = training_args.weight_decay

    return training_args, optimizer_args

def get_pretraining_name_and_tags(args):
    args.wandb_name = f"{args.wandb_name_prefix}-" if args.wandb_name_prefix else ""
    args.wandb_name += f"opt-{args.optimizer}-dtype-{args.dtype}-amp-{int(args.amp)}-bs-{args.total_batch_size}-sch-{args.scheduler}-wd-{args.weight_decay}-lr-{args.lr}"
    args.wandb_tags = [args.optimizer, args.dtype, 'amp' if args.amp else 'no-amp', 'bs_' + str(args.total_batch_size), 'sch_' + args.scheduler, 'wd_' + str(args.weight_decay), 'lr_'+str(args.lr)]
    if args.grad_clipping != 0.0:
        args.wandb_name += f"-clip-{args.grad_clipping}"
        args.wandb_tags += [f"clip-{args.grad_clipping}"]

    # TODO adam and other
    if "sgd" in args.optimizer:
        args.wandb_name += f"-nesterov-{int(args.nesterov)}-momentum-{args.beta1}-sign-{int(args.sgd_sign_update)}"
        args.wandb_tags += ['nesterov' if args.reset_statistics else 'no-nesterov', f"momentum_{args.beta1}", 'sign' if args.sgd_sign_update else 'no-sign']

    if any(keyword in args.optimizer for keyword in ["galore", "coord", "block"]):
        args.wandb_name += f"-{args.proj_params_lr_scale}-gap-{args.update_gap}-dens-{args.density}-reset-{int(args.reset_statistics)}-inactive-{args.inactive_update_rule}-{args.inactive_lr_scale}-"
    if any(proj_type in args.optimizer for proj_type in ['galore', 'coord', 'block']):
        args.wandb_tags += ['proj', 'gap_' + str(args.update_gap), 'dens' + str(args.density), 'reset' if args.reset_statistics else 'no-reset', 'inactive_' + args.inactive_update_rule]
        args.wandb_tags += ['inactive-lr_' + str(args.inactive_lr_scale)] if args.inactive_lr_scale != 1.0 else [] + ['proj-lr_' + str(args.proj_params_lr_scale)] if args.proj_params_lr_scale != 1.0 else []
    
    if "galore" in args.optimizer:
        args.wandb_name += f"galore-{args.proj_type}-{args.proj_side}"
    elif "coord" in args.optimizer:
        args.wandb_name += f"coord-{args.coord_choice}"
    elif "block" in args.optimizer:
        args.wandb_name += f"block-{args.block_order}"
        args.wandb_name += f"-emb-{int(args.proj_embeds)}-logit-{int(args.proj_logits)}-norm-{int(args.proj_norms)}"
    if 'galore' in args.optimizer:
        args.wandb_tags += ['galore', args.proj_type] + [] if args.proj_side == "std" else ['proj-side_' + args.proj_side]
    elif 'coord' in args.optimizer:
        args.wandb_tags += ['coord', args.coord_choice]
    elif "block" in args.optimizer:
        args.wandb_tags += ['block', f"block-{args.block_order}"]

    args.wandb_name += f"-seed-{args.seed}"
    args.wandb_tags += ['seed_' + str(args.seed)]

def get_finetuning_name_and_tags(training_args, optimizer_args):
    training_args.run_name = ""
    sch = training_args.lr_scheduler_type.split(".")[0] # TODO
    training_args.run_name += f"opt-{optimizer_args.optimizer}-amp-{int(training_args.bf16) or int(training_args.fp16)}-bs-{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}-sch-{sch}-wd-{training_args.weight_decay}-lr-{training_args.learning_rate}"
    
    per_device_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    training_args.wandb_tags = [optimizer_args.optimizer, 'amp' if training_args.bf16 or training_args.fp16 else 'no-amp', 'bs_' + str(per_device_batch_size), 'sch_' + sch, 'wd_' + str(training_args.weight_decay), 'lr_'+str(training_args.learning_rate)]
    
    if training_args.max_grad_norm != 0.0:
        training_args.run_name += f"-clip-{training_args.max_grad_norm}"
        training_args.wandb_tags += [f"clip-{training_args.max_grad_norm}"]

    if "sgd" in optimizer_args.optimizer:
        training_args.run_name += f"-nesterov-{int(optimizer_args.nesterov)}-momentum-{optimizer_args.beta1}-sign-{int(optimizer_args.sgd_sign_update)}"
        training_args.wandb_tags += [
            'nesterov' if optimizer_args.reset_statistics else 'no-nesterov',
            f"momentum_{optimizer_args.beta1}",
            'sign' if optimizer_args.sgd_sign_update else 'no-sign'
        ]

    if any(keyword in optimizer_args.optimizer for keyword in ["galore", "coord", "block"]):
        training_args.run_name += f"-{optimizer_args.proj_params_lr_scale}-gap-{optimizer_args.update_gap}-rank-{optimizer_args.rank}-reset-{int(optimizer_args.reset_statistics)}-inactive-{optimizer_args.inactive_update_rule}-{optimizer_args.inactive_lr_scale}-"

    if any(proj_type in optimizer_args.optimizer for proj_type in ['galore', 'coord', 'block']):
        training_args.wandb_tags += [
            'proj', 
            'gap_' + str(optimizer_args.update_gap), 
            'rank-' + str(optimizer_args.rank), 
            'reset' if optimizer_args.reset_statistics else 'no-reset', 
            'inactive_' + optimizer_args.inactive_update_rule
        ]
        
        if optimizer_args.inactive_lr_scale != 1.0:
            training_args.wandb_tags += ['inactive-lr_' + str(optimizer_args.inactive_lr_scale)]
            
        if optimizer_args.proj_params_lr_scale != 1.0:
            training_args.wandb_tags += ['proj-lr_' + str(optimizer_args.proj_params_lr_scale)]

    if "galore" in optimizer_args.optimizer:
        training_args.run_name += f"galore-{optimizer_args.proj_type}-{optimizer_args.proj_side}"
    elif "coord" in optimizer_args.optimizer:
        training_args.run_name += f"coord-{optimizer_args.coord_choice}"
    elif "block" in optimizer_args.optimizer:
        training_args.run_name += f"block-{optimizer_args.block_order}"

    if 'galore' in optimizer_args.optimizer:
        training_args.wandb_tags += ['galore', optimizer_args.proj_type] 
        if optimizer_args.proj_side != "std":
            training_args.wandb_tags += ['proj-side_' + optimizer_args.proj_side]
            
    elif 'coord' in optimizer_args.optimizer:
        training_args.wandb_tags += ['coord', optimizer_args.coord_choice]
            
    elif "block" in optimizer_args.optimizer:
        training_args.wandb_tags += ['block', f"block-{optimizer_args.block_order}"]
    
    if optimizer_args.lora_enabled:
        training_args.run_name += f"-lora-{optimizer_args.rank}-{optimizer_args.lora_alpha}-{optimizer_args.lora_dropout}"
        training_args.wandb_tags += ["lora"]
        training_args.run_name += f"-all" if optimizer_args.lora_all_modules else f"-qv"
    training_args.run_name += f"-freeze_emb-{int(optimizer_args.freeze_embeddings)}"
    training_args.run_name += f"-freeze_norm-{int(optimizer_args.freeze_norms)}"

    training_args.run_name += f"-seed-{training_args.seed}"
    training_args.wandb_tags += ['seed_' + str(training_args.seed)]