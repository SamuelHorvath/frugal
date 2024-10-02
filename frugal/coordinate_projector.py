import torch

class CoordinateProjector:
    def __init__(self, density, grad_shape, coord_choice="columns"):
        self.density = density
        self.coord_choice = coord_choice
        self.grad_shape = grad_shape
        self.indices = None

    def update_proj(self, grad):

        if self.coord_choice == "columns":
            self.indices = self._get_indices(self.grad_shape[1], grad.device)
        elif self.coord_choice == "rows":
            self.indices = self._get_indices(self.grad_shape[0], grad.device)
        else:
            self.indices = self._get_indices(self.grad_shape[0]*self.grad_shape[1], grad.device)
            self.indices = (
                (self.indices // self.grad_shape[1]),
                (self.indices % self.grad_shape[1])
            )

    def project_down(self, full_rank):
        
        if self.coord_choice == "columns":
            low_rank = full_rank[:, self.indices]
        elif self.coord_choice == "rows":
            low_rank = full_rank[self.indices]
        else:
            low_rank = full_rank[self.indices[0], self.indices[1]]

        return low_rank

    def project_up(self, low_rank, full_rank=None):
        
        if full_rank is None:
            full_rank = torch.zeros(size=self.grad_shape, dtype=low_rank.dtype, device=low_rank.device)
        
        if self.coord_choice == "columns":
            full_rank[:, self.indices] = low_rank.to(full_rank.device, full_rank.dtype)
        elif self.coord_choice == "rows":
            full_rank[self.indices] = low_rank.to(full_rank.device, full_rank.dtype)
        else:
            full_rank[self.indices[0], self.indices[1]] = low_rank.to(full_rank.device, full_rank.dtype)
        
        return full_rank
        
    def _get_indices(self, total_length, device):
        return torch.randperm(total_length, device=device)[:int(total_length * self.density)]