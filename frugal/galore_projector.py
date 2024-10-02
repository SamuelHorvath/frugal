import torch

class GaLoreProjector:
    def __init__(self, density, grad_shape, proj_side='std', proj_type="svd", verbose=False):
        self.verbose = verbose
        self.ortho_matrix = None
        if (proj_side == 'right' or 
            (proj_side == 'std' and grad_shape[0] >= grad_shape[1]) or 
            (proj_side == 'reverse_std' and grad_shape[0] < grad_shape[1])):
            self.proj_side = "right"
            self.rank = round(grad_shape[1] * density)
        elif (proj_side == 'left' or 
            (proj_side == 'reverse_std' and grad_shape[0] >= grad_shape[1]) or
            (proj_side == 'std' and grad_shape[0] < grad_shape[1])):
            self.proj_side = "left"
            self.rank = round(grad_shape[0] * density)
        else:
            self.proj_side = "full"
            self.rank = round(min(grad_shape) * (density))
        self.proj_type = proj_type

    def update_proj(self, full_rank_grad):
        if self.proj_type == "svd":
            self.ortho_matrix = self._get_orthogonal_matrix(full_rank_grad, self.rank, type=self.proj_side)
        elif self.proj_type == "random":
            self.ortho_matrix = self._get_random_orthogonal_matrix(full_rank_grad, self.rank, type=self.proj_side)
        elif self.proj_type == "randperm":
            self.ortho_matrix = self._get_randperm_matrix(full_rank_grad, self.rank, type=self.proj_side)
        else:
            raise NameError("Wrong proj_type.")

    def project_down(self, full_rank):
        
        if self.proj_side == "right":
            low_rank = torch.matmul(full_rank, self.ortho_matrix.t())
        elif self.proj_side == "left":
            low_rank = torch.matmul(self.ortho_matrix.t(), full_rank)
        else:
            low_rank = torch.matmul(torch.matmul(self.ortho_matrix[0].t(), full_rank), self.ortho_matrix[1].t())

        return low_rank

    def project_up(self, low_rank):
        
        if self.proj_side == 'right':
            full_rank = torch.matmul(low_rank, self.ortho_matrix)
        elif self.proj_side == 'left':
            full_rank = torch.matmul(self.ortho_matrix, low_rank)
        else:
            full_rank = torch.matmul(torch.matmul(self.ortho_matrix[0], low_rank), self.ortho_matrix[1])
        
        return full_rank
        
    # svd decomposition
    def _get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        if type=='right':
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]

    # random projection
    def _get_random_orthogonal_matrix(self, weights, rank, type):
        original_device = weights.data.device
        if weights.data.dtype != torch.float:
            float_data = False
            original_type = weights.data.dtype    
        else:
            float_data = True
        
        if type=='right':
            Vh = torch.nn.init.orthogonal_(torch.empty(rank, weights.shape[1], device=original_device))
            B = Vh
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            U = torch.nn.init.orthogonal_(torch.empty(weights.shape[0], rank, device=original_device))
            A = U
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            U = torch.nn.init.orthogonal_(torch.empty(weights.shape[0], rank, device=original_device))
            Vh = torch.nn.init.orthogonal_(torch.empty(rank, weights.shape[1], device=original_device))
            A = U
            B = Vh
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
    
    # randperm
    def _get_randperm_matrix(self, weights, rank, type):
        original_device = weights.data.device
        if weights.data.dtype != torch.float:
            float_data = False
            original_type = weights.data.dtype    
        else:
            float_data = True
        
        if type=='right':
            Vh = torch.zeros(weights.shape[1], weights.shape[1], device=original_device)
            Vh[torch.arange(weights.shape[1]), torch.randperm(weights.shape[1])] = 1.0
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            U = torch.zeros(weights.shape[0], weights.shape[0], device=original_device)
            U[torch.arange(weights.shape[0]), torch.randperm(weights.shape[0])] = 1.0
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            U = torch.zeros(weights.shape[0], weights.shape[0], device=original_device)
            U[torch.arange(weights.shape[0]), torch.randperm(weights.shape[0])] = 1.0
            Vh = torch.zeros(weights.shape[1], weights.shape[1], device=original_device)
            Vh[torch.arange(weights.shape[1]), torch.randperm(weights.shape[1])] = 1.0
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]