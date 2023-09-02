import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import itertools
import numpy as np

from torchsummary import summary
from tqdm.auto import tqdm

from . import utils

def downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)

def upsample(dim_in, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim_in, dim_out, 3, padding=1))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def extract(a, t, x_shape):
    a = a.to(t.device)
    batch, *rest = t.shape
    out = a.gather(-1, t) # out has the same size of t ('batch' elements)
    return out.reshape(batch, *((1,) * (len(x_shape)-1)))

def unnormalized(t):
    return (t + 1) * 0.5

def normalized(t):
    return t * 2 - 1

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var  = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.alpha + self.beta

class SinPosEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # x.shape : [m], emb.shape: [n]
        # x[:,None].shape = [m,1], emb[None,:].shape = [1,n]
        # shape(x[:,None] * emb[None,:]): [m,n]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.acti = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.acti(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim_out * 2))

        self.block1  = Block(dim, dim_out, groups=groups)
        self.block2  = Block(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.resconv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.resconv = nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = time_emb.view(*time_emb.shape, 1, 1)
        scale_shift = time_emb.chunk(2, dim = 1)
        
        h = self.block1(x, scale_shift)
        h = self.block2(h)

        return h + self.resconv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim  = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3,dim=1)

        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)

        q = torch.softmax(q, dim=-2)
        k = torch.softmax(k, dim=-1)

        q = q * self.scale
        v = v / (h * w)
        v = v.permute((0,1,3,2))
        
        context = torch.matmul(k,v)
        context = context.permute((0,1,3,2))
        #context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        
        out = torch.matmul(context, q)
        out = out.view(b, -1, h, w)
        
        return  self.to_out(out)

class WithLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(self, img_ch, init_dim):
        super().__init__()
        self.img_ch  = img_ch
        self.init_dim  = init_dim
        self.init_conv = nn.Conv2d(img_ch, init_dim, 7, padding=3)

        # Time embedding
        time_emb_dim = self.init_dim * 4
        self.time_mlp = nn.Sequential(
            SinPosEmbedding(self.init_dim),
            nn.Linear(self.init_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Conv Layers
        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        scales = (1, 1, 2, 4, 8)

        for i in range(len(scales) - 1):
            dim_in  = self.init_dim * scales[i]
            dim_out = self.init_dim * scales[i + 1]
            is_last = i == (len(scales) - 2)
            
            self.down_layers.append(nn.ModuleList([
                ResNetBlock(dim_in, dim_in, time_emb_dim=time_emb_dim),
                Residual(WithLayerNorm(dim_in, LinearAttention(dim_in))),
                downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = scales[-1] * self.init_dim
        self.mid_block = ResNetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim)

        for i in reversed(range(len(scales) - 1)):
            dim_in  = self.init_dim * scales[i]
            dim_out = self.init_dim * scales[i + 1]
            is_last = i == 0
            self.up_layers.append(nn.ModuleList([
                ResNetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_emb_dim),
                Residual(WithLayerNorm(dim_out, LinearAttention(dim_out))),
                upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResNetBlock(self.init_dim * 2, self.init_dim, time_emb_dim=time_emb_dim)
        self.final_conv = nn.Conv2d(self.init_dim, self.img_ch, 1)
    
    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        h = []

        for block, attn, down in self.down_layers:
            x = block(x, t)
            x = attn(x)
            h.append(x)
            x = down(x)
        
        x = self.mid_block(x, t)

        
        for block, attn, up in self.up_layers:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block(x, t)
            x = attn(x)
            x = up(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        img_ch,
        img_size,
        init_dim,
        timesteps,
        device
    ):
        super().__init__()
        self.device = device
        self.img_ch = img_ch
        self.img_size = img_size
        self.loss_fn = F.l1_loss
        self.model = Unet(img_ch, init_dim)

        # Coeficients
        # =====================================================================================
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
    
        self.num_timesteps = int(timesteps)
        
        self.sampling_timesteps = timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_samplint_eta = 1.

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('sqrt_alphas', torch.sqrt(alphas))
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Diffusion q(x_t | x_{t-1}):  x_{t-1} ---> x_t
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))

        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior: q(x_{t-1} | x_t, x_0)
        # =====================================================================================

        # Posterior Mean: \hat{\mu}_t(x_t, x_0)
        register_buffer('posterior_mean_coef1', 
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', 
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))      

        # Posterior Variance: \hat{\beta}_t
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min = 1e-20)))
        
        # How t is designed?
        #
        # t = torch.randint(0, self.num_timesteps, (batch,), device=device).long()
        # t: [--- batch elements ---], each element from 0 to self.num_timesteps - 1
        # batch: number of images on each batch

    def predict_start_from_noise(self, x_t, t, noise):
        """
            Eq. 4 from the DDPM paper: 
                
                x_t = \sqrt{\hat{alpha_t}} * x_0 + \sqrt{1 - \hat{alpha_t}} * noise
        
            ==> x_0 = (x_t - \sqrt{1 - \hat{alpha_t}} * noise) / \sqrt{\hat{alpha_t}}
                x_0 = sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
        """

        # Return the value of 'sqrt_recip_alphas_cumprod' at index 't' with the shape of 'x_t'
        # Return the value of 'sqrt_recipm1_alphas_cumprod' at index 't' with the shape of 'x_t'
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
            Eq. 4 from the DDPM paper: 
                
                x_t = \sqrt{\hat{alpha_t}} * x_0 + \sqrt{1 - \hat{alpha_t}} * noise
            ==> noise = (x_t - \sqrt{\hat{alpha_t}} * x_0) / \sqrt{1 - \hat{alpha_t}}
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /\
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """
            Eq. 7 from the DDPM paper.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        noise_pred = self.model(x, t)
        x_start = self.predict_start_from_noise(x, t, noise_pred)

        return noise_pred, x_start

    def p_mean_variance(self, x, t, clip_denoised):
        noise_pred, x_start = self.model_predictions(x, t)

        x_start = torch.clamp(x_start, -1., 1.) if clip_denoised else x_start

        mean, _, log_var = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return mean, log_var

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        """
            Sampling Algorithm: STEP 3,4
        """

        # Step 3: z_noise ~ N(0, I) if t > 0 else 0.
        z_noise = torch.randn_like(x) if t > 0 else 0.

        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        # Step 4: \mu_t, \sigma_t = q(x_{t-1} | x_t, x_0)
        #         x_0 = \frac{1}{\sqrt{\hat{alpha_t}}} * x_t - \sqrt{\frac{1 - \hat{alpha_t}}{\hat{alpha_t}}} * noise_pred(x_t)
        #         Our model must predict noise from x_t to x_0 (Eq. 4)
        mean, log_var = self.p_mean_variance(x=x, t=t_batch, clip_denoised=clip_denoised)
        
        # Step 4: x_{t-1} = \mu_t + \sigma_t * z_noise
        return mean + (0.5 * log_var).exp() * z_noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
            Sampling Algorithm: STEP 1,2
        """
        # Step 1: x_0 ~ N(0, I)
        img = torch.randn(*shape, device=self.device)   

        # Step 2: for t = T, T-1, ..., 1 do
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step'):
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        times = torch.linspace(0., self.num_timesteps, steps = self.sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = self.betas.device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full(
                (shape[0],), time, device = self.betas.device, dtype = torch.long)

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalized(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, self.img_ch, self.img_size, self.img_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        """
            Eq. 4 from the DDPM paper: 
                
                x_t = \sqrt{\hat{alpha_t}} * x_0 + \sqrt{1 - \hat{alpha_t}} * noise
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, img):
        """
            Training Algorithm: STEP 2,3,4,5
        """
        b, c, h, w = img.shape
        assert h == self.img_size and w == self.img_size
        
        # Step 2, 3, 4   
        t   = torch.randint(0, self.num_timesteps, (b,), device=img.device).long()
        img = normalized(img)
        noise = torch.randn_like(img)
        
        x = self.q_sample(x_start = img, t = t, noise = noise)

        # Our model must predict noise from x_t to x_0 (Eq. 4)
        model_out = self.model(x, t)

        # Step 5
        loss = self.loss_fn(model_out, noise, reduction = 'none')
        loss = torch.mean(loss)

        return loss

class Trainer():
    def __init__(self, config):
        self.loader_dict = utils.get_celeb_data(
            img_size=config.data.img_size, batch_size=config.hyp.batch_size)
        
        self.train_load = self.loader_dict['train_loader'][0]
        self.train_size = self.loader_dict['train_loader'][1]
        self.valid_load = self.loader_dict['valid_loader'][0]

        checkpoints_path = os.path.join(
                config.exp.save_dir,
                config.exp.name,
                'try-%d' % config.exp.try_num,
                'checkpoints')

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.device = config.exp.device
        self.lr     = config.hyp.lr

        self.checkpoints_path = checkpoints_path
        #self.config = config
        self.beta1  = config.hyp.beta1
        self.beta2  = config.hyp.beta2
        self.img_size = config.data.img_size
        self.batch_size = config.hyp.batch_size
        self.load_epoch = config.exp.load_epoch
        self.save_freq = config.exp.save_freq
        self.epochs = config.hyp.epochs
        self.timesteps = config.hyp.timesteps

    def __set_model(self):
        self.model = GaussianDiffusion(
            img_ch=3,
            img_size=self.img_size,
            init_dim=64,
            timesteps=self.timesteps,
            device=self.device).to(self.device)
        self.opt_model = torch.optim.Adam(
            itertools.chain(self.model.parameters()), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        summary(self.model, (3, self.img_size, self.img_size))   
        self.__load_model()

    def show_inputs(self):
        plt.ion()
        fig = plt.figure(figsize=(5, 8), dpi=80)
        
        for image_batch, _ in self.train_load:
            utils.save_batch(image_batch, "inputs.png")

    def sample_batch(self):
        self.__set_model()
        self.model.eval()
        x_hat = self.model.sample(self.batch_size).cpu().detach()
        utils.save_batch(x_hat, "sample.png")    

    def train_step(self):
        self.__set_model()

        x_hat = self.model.sample(int(self.batch_size)).cpu().detach()
        utils.save_batch(x_hat, f"batch-{0}.png") 

        for ep in range(self.load_epoch + 1, self.epochs + 1):
            start_t = time.time()
            self.model.train()
            train_loss = []

            it = 0
            for image_batch, _ in self.train_load:
                loss = self.optimize_parameters(x=image_batch.to(self.device))

                it += image_batch.size(0)

                train_loss.append(loss)
                print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f}", end="")

            train_loss_mean = np.mean(train_loss)
            end_t = time.time()   
            print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f} in {end_t-start_t:.1f} sec.")
            
            self.model.eval()
            # Show sample
            x_hat = self.model.sample(int(self.batch_size)).cpu().detach()
            utils.save_batch(x_hat, f"batch-{ep}.png")  

            if ep % self.save_freq == 0: 
                valid_loss = []

                for image_batch_v, _ in self.valid_load:
                    loss = self.model(image_batch.to(self.device))
                    valid_loss.append(loss.item())
                    #break
                valid_loss_mean = np.mean(valid_loss)   

                print(f"\tloss(val)={valid_loss_mean:.3f}")
                self.__save_model(ep)

    def __save_model(self, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['state_dict_net'] = self.model.state_dict()   
        checkpoint['opt_model'] = self.opt_model.state_dict()

        torch.save(checkpoint, save_path)
        print("Done.")

    def __load_model(self):
        if self.load_epoch <= 0:
            return
        print('\nLoading "model-{:d}"...'.format(self.load_epoch), end='')
        file_model = 'model-{:d}.pth'.format(self.load_epoch)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['state_dict_net'])
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        #self.model.set_actnorm_init()

        print("Done.")

    def optimize_parameters(self, x):

        loss = self.model(x)
        self.opt_model.zero_grad()
        loss.backward()
        self.opt_model.step()

        return loss.item()
