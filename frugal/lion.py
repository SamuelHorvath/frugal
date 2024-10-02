from typing import Callable, Iterable, Tuple, Dict
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .proj_optimizer_templates import GaloreOptimizer, CoordOptimizer, BlockOptimizer

class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _is_state_empty(self, state):
        return any(key not in state for key in ["step", "exp_avg"])

    @torch.no_grad()
    def _init_state(self, example=None, state=None):
        assert isinstance(state, Dict) or state is None
        assert isinstance(example, torch.Tensor) or example is None
        assert not (state is None and example is None), "One of the arguments `state` and `example` should be specified."
        if state is not None and not self._is_state_empty(state):
            state["step"] = 0
            state["exp_avg"].zero_()
        else:
            if state is None:
                state = {}
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(example)
        return state

    @torch.no_grad()
    def _compute_update(self, grad, state, lr, betas, **kwargs):

        beta1, beta2 = betas
        update = state["exp_avg"] * beta1 + grad * (1 - beta1)
        state["exp_avg"].mul_(beta2).add_(grad, alpha=1 - beta2)

        return update.sign_() * (-lr)

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

                update = self._compute_update(grad, state, group["lr"], group["betas"])

                p.add_(update)
        
        return loss

class GaloreLion(GaloreOptimizer, Lion):

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

        # lion specific
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.00,
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
        Lion.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
        )


class CoordLion(CoordOptimizer, Lion):

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

        # lion specific
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.00,
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
        Lion.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
        )


class BlockLion(BlockOptimizer, Lion):

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

        # lion specific
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.00,
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
        Lion.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
        )