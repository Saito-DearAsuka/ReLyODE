import torch
import torch.nn as nn
import torch.nn.functional as F
# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        # n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


####################################  model  V  #################################################
class ICNN(nn.Module):
    """Input Convex Neural Network for Lyapunov function"""

    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(ICNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Input weights (W) - using Conv2d
        self.W = nn.ModuleList([
            nn.Conv2d(input_dim, hidden_dims[0], kernel_size=3, padding=1, bias=True)
        ])

        # Hidden weights (W)
        for i in range(len(hidden_dims) - 1):
            self.W.append(nn.Conv2d(input_dim, hidden_dims[i + 1], kernel_size=3, padding=1, bias=True))

        # Coupling weights (U) - must be positive
        self.U = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            U_layer = nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, padding=1, bias=False)
            U_layer.weight.data.uniform_(0, 0.1)
            self.U.append(U_layer)

        # Final layers
        self.final_W = nn.Conv2d(input_dim, 1, kernel_size=3, padding=1, bias=True)
        self.final_U = nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1, bias=False)
        self.final_U.weight.data.uniform_(0, 0.1)

    def forward(self, x):
        z = F.softplus(self.W[0](x))

        for i in range(len(self.hidden_dims) - 1):
            z = F.softplus(self.W[i + 1](x) + self.U[i](z))

        # Ensure V(0) = 0
        z_zero = torch.zeros_like(x)
        v_zero = self.final_W(z_zero) + self.final_U(F.softplus(self.W[-1](z_zero)))

        # Add quadratic term for strict positive definiteness
        output = self.final_W(x) + self.final_U(z) - v_zero + 0.1 * torch.sum(x ** 2, dim=1, keepdim=True)
        return output



class ODEFunc(nn.Module):
    def __init__(self, opt, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.opt = opt

        # Original dynamics network
        self.gradient_net = ode_func_net

        # Lyapunov network
        self.lyap_net = ICNN(input_dim).to(device)

        # Stability parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def get_lyap_derivative(self, x, f_x):
        """Compute derivative of Lyapunov function along trajectories"""
        had_no_grad = not torch.is_grad_enabled()
        if had_no_grad:
            torch.set_grad_enabled(True)

        # clone x and set requires_grad=True
        x_grad = x.detach().clone()
        x_grad.requires_grad_(True)

        # compute V and its derivative
        try:
            V = self.lyap_net(x_grad)
            grad_V = torch.autograd.grad(V.sum(), x_grad, create_graph=True)[0]
            lyap_derivative = torch.sum(grad_V * f_x, dim=1, keepdim=True)
        finally:
            if had_no_grad:
                torch.set_grad_enabled(False)

        return lyap_derivative, grad_V

    def project_stable(self, x, f_hat, grad_V):
        """Project dynamics onto stable set"""
        had_no_grad = not torch.is_grad_enabled()
        if had_no_grad:
            torch.set_grad_enabled(True)

        try:
            x_grad = x.detach().clone()
            x_grad.requires_grad_(True)

            V = self.lyap_net(x_grad)

            grad_V_norm = torch.norm(grad_V, dim=1, keepdim=True).clamp(min=1e-6)
            inner_prod = torch.sum(grad_V * f_hat, dim=1, keepdim=True)

            # V_point_wise_loss = F.relu(inner_prod+self.alpha*V)
            projection_term = (F.relu(inner_prod + self.alpha * V) /
                               (grad_V_norm ** 2)) * grad_V

            f = f_hat - projection_term
        finally:
            if had_no_grad:
                torch.set_grad_enabled(False)

        return f

    def forward(self, t_local, y):
        """Perform one step in solving ODE"""
        # Get nominal dynamics
        f_hat = self.gradient_net(y)

        # Get Lyapunov derivative and gradient
        lyap_deriv, grad_V = self.get_lyap_derivative(y, f_hat)

        # Project onto stable dynamics
        f = self.project_stable(y, f_hat, grad_V)

        return f

    def sample_next_point_from_prior(self, t_local, y):
        return self.forward(t_local, y)

    def compute_total_loss(self, x):
        had_no_grad = not torch.is_grad_enabled()
        if had_no_grad:
            torch.set_grad_enabled(True)

        try:
            x_grad = x.detach().clone()
            x_grad.requires_grad_(True)

            f_x = self.gradient_net(x_grad)
            # 计算 Lyapunov loss
            lyap_loss = self.lyap_net.compute_lyapunov_loss(x_grad, f_x)
        finally:
            if had_no_grad:
                torch.set_grad_enabled(False)
        return lyap_loss


