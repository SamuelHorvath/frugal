# based on https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
# _single_gpu realization

from typing import Callable, Iterable, Tuple, Dict

import torch
from torch import nn
from torch.optim import Optimizer

from .proj_optimizer_templates import GaloreOptimizer, CoordOptimizer, BlockOptimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sign_update=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, sign_update=sign_update)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def _init_state(self, example, state=None):
        assert isinstance(example, torch.Tensor)
        assert isinstance(state, Dict) or state is None
        if state is None:
            state = {}
        state["step"] = 0
        state["momentum_buffer"] = torch.clone(example).detach()
        return state

    @torch.no_grad()
    def _compute_update(self, grad, state, lr, momentum, nesterov, dampening, sign_update, **kwargs):

        if momentum != 0:
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        
        if sign_update:
            grad = grad.sign()
        
        return grad * (-lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                if len(state) == 0:
                    self._init_state(example=p, state=state)
                    if not group['momentum']:
                        state.pop("momentum_buffer", None)
                
                state["step"] += 1

                update = self._compute_update(grad, state, group["lr"], group["momentum"], group["nesterov"], group["dampening"], group["sign_update"])

                p.add_(update)

        return loss


class GaloreSGD(GaloreOptimizer, SGD):

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

        _example_state_init=True,

        # galore specific
        proj_side='std',
        proj_type='svd',

        # sgd specific
        lr: float = 1e-3,
        momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sign_update=False,
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
            proj_type=proj_type
        )
        SGD.__init__(
            self, params, 
            lr=lr, 
            momentum=momentum, 
            dampening=dampening, 
            weight_decay=weight_decay, 
            nesterov=nesterov, 
            sign_update=sign_update
        )


class CoordSGD(CoordOptimizer, SGD):

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

        _example_state_init=True,

        # coord specific
        coord_choice='columns',

        # sgd specific
        lr: float = 1e-3,
        momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sign_update=False,
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
        SGD.__init__(
            self, params, 
            lr=lr, 
            momentum=momentum, 
            dampening=dampening, 
            weight_decay=weight_decay, 
            nesterov=nesterov, 
            sign_update=sign_update
        )


class BlockSGD(BlockOptimizer, SGD):

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

        _example_state_init=True,

        # block specific
        block_order='random',

        # sgd specific
        lr: float = 1e-3,
        momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sign_update=False,
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
        SGD.__init__(
            self, params, 
            lr=lr, 
            momentum=momentum, 
            dampening=dampening, 
            weight_decay=weight_decay, 
            nesterov=nesterov, 
            sign_update=sign_update
        )