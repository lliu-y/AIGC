import torch.nn as nn
import torch
import torch.nn.functional as F
import math

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(0, t)
    return out.view(b, *((1,) * (len(x_shape) - 1)))  # Return output in the correct shape

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, predict, target, weighted=1.0):
        """
        predict, target: [batch_size, action_dim]
        """
        loss = self._loss(predict, target)
        WeightedLoss = (loss * weighted).mean()
        return WeightedLoss

class WeightedL1(WeightedLoss):
    def _loss(self, predict, target):
        return torch.abs(predict - target)

class WeightedL2(WeightedLoss):
    def _loss(self, predict, target):
        return F.mse_loss(predict, target, reduction='mean')

class SinusoidalPosEmb(nn.Module):
    def __init__(self, t_dim):
        super().__init__()
        self.t_dim = t_dim

    def forward(self, x):
        device = x.device
        half_dim = self.t_dim // 2

        # Compute the scaling factor for position encoding
        emb = -math.log(10000) / (half_dim - 1)
        emb = torch.arange(half_dim, device=device) * emb

        # Calculate the position encoding
        emb = x[:, None] * emb[None, :]  # Ensure x is a 1D tensor of time steps

        # Concatenate the sin and cos encodings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, time_dim=16):
        super(MLP, self).__init__()

        self.time_dim = time_dim
        self.action_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim * 2, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_dim = state_dim + action_dim + time_dim * 2

        self.mid_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.finnal_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        time = self.time_mlp(time)
        time = time.view(time.size(0), -1)  # 确保 time 的形状正确
        x = torch.cat([x, state, time], dim=1)
        x = self.mid_layer(x)
        x = self.finnal_layer(x)
        return x

Losses = {
    "l1": WeightedL1(),
    "l2": WeightedL2(),
}

class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, predict_epsion=True, **kwargs):
        super(Diffusion, self).__init__()

        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.T = kwargs['T']
        self.clip_denoised = clip_denoised
        self.predict_epsion = predict_epsion
        self.device = torch.device(kwargs['device'])
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas

        alphas_cumprod = torch.cumprod(alphas, axis=0)

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alpha_cimprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recip_alpha_cimprod_m1", torch.sqrt(1.0 / alphas_cumprod - 1))

        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]
        self.to(self.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, state, noise=None):
        noise = noise or torch.randn_like(x_start)
        x = self.q_sample(x_start=x_start, t=t)
        t = t.unsqueeze(-1)
        predict_noise = self.model(x, t, state)
        return self.loss_fn(predict_noise, noise)

    def predict_start_from_noise(self, x, t, predict_noise):
        return (
            extract(self.sqrt_recip_alpha_cimprod, t, x.shape) * x
            - extract(self.sqrt_recip_alpha_cimprod_m1, t, x.shape) * predict_noise
        )

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape)
        )

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, t, state):
        predict_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, predict_noise)
        x_recon.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, state):
        b = x.shape[0]
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x, device=self.device)

        none_zero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + none_zero_mask * noise * (0.5 * torch.exp(model_log_variance))

    def p_sample_loop(self, state, shape, *args, **kwargs):
        batch_size = state.shape[0]
        x = torch.randn(shape, device=self.device, requires_grad=False)

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size, 1), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, state)

        return x

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]

        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp(-1, 1)

    def training_step(self, batch, batch_idx):
        state, x_start = batch
        t = torch.randint(0, self.T, (state.shape[0],), device=self.device).long()

        x_t = self.q_sample(x_start, t)

        model_mean, model_log_variance = self.p_mean_variance(x_t, t, state)
        loss = self.loss_fn(model_mean, x_start)

        return loss
