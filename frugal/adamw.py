import math
import warnings
from typing import Callable, Iterable, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.optimizer import _dispatch_sqrt

from transformers.utils.versions import require_version

from .proj_optimizer_templates import GaloreOptimizer, CoordOptimizer, BlockOptimizer

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    def _is_state_empty(self, state):
        return any(key not in state for key in ["step", "exp_avg", "exp_avg_sq"])

    @torch.no_grad()
    def _init_state(self, example=None, state=None):
        assert isinstance(state, Dict) or state is None
        assert isinstance(example, torch.Tensor) or example is None
        assert not (state is None and example is None), "One of the arguments `state` and `example` should be specified."
        if state is not None and not self._is_state_empty(state):
            state["step"] = 0
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
        else:
            if state is None:
                state = {}
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(example)
            state["exp_avg_sq"] = torch.zeros_like(example)
        return state

    @torch.no_grad()
    def _compute_update(self, grad, state, lr, betas, eps, correct_bias, **kwargs):
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = betas

        exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt()

        step_size = lr
        if correct_bias:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            denom.div_(bias_correction2_sqrt)
        
        denom.add_(eps)
        
        return exp_avg / denom * (-step_size)


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
            
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(example=p, state=state)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                state["step"] += 1

                update = self._compute_update(grad, state, group["lr"], group["betas"], group["eps"], group["correct_bias"])

                p.add_(update)
        
        return loss


class GaloreAdamW(GaloreOptimizer, AdamW):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # galore specific
        proj_side='std',
        proj_type='svd',

        # adam specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            proj_side=proj_side,
            proj_type=proj_type,
        )
        AdamW.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )


class CoordAdamW(CoordOptimizer, AdamW):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # coord specific
        coord_choice='columns',

        # adam specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            coord_choice=coord_choice,
        )
        AdamW.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )


class BlockAdamW(BlockOptimizer, AdamW):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # block specific
        block_order='random',

        # adam specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            block_order=block_order,
        )
        AdamW.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )