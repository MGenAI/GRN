import torch
import torch.nn as nn
from grn.models.grn_c2i import GRN_models

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        vae_down_sample = 16
        self.net = GRN_models[args.model](
            input_size=args.img_size // vae_down_sample,
            in_channels=args.in_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            args=args,
        )
        self.img_size = args.img_size // vae_down_sample
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.args = args

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        # x shape: [B,d,h,w]
        if self.args.method == 'GRN_ind':
            classes = 2**self.args.hbq_round
        elif self.args.method == 'GRN_bit':
            classes = 2
        random_labels = torch.randint(0, classes, size=x.shape, device=x.device)
        x_mask = torch.rand(size=x.shape, device=x.device) < t
        z = torch.where(x_mask, x, random_labels)
        x_pred = self.net(z, t.flatten(), labels_dropped) # x_pred shape: [B, classes, d, h, w]
        # ce loss
        gt_labels = x # [B,d,h,w]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            loss = torch.nn.functional.cross_entropy(x_pred, gt_labels)
        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        if self.args.method == ['GRN_ind']:
            classes = 2 ** self.args.hbq_round
            rand_labels = torch.randint(0, classes, (bsz, self.net.in_channels//classes, self.img_size, self.img_size), device=device)
            z = rand_labels
        elif self.args.method in ['GRN_bit']:
            classes = 2
            rand_labels = torch.randint(0, classes, (bsz, self.net.in_channels//classes, self.img_size, self.img_size), device=device)
            z = rand_labels
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        for i in range(self.steps): # self.steps=50
            t = timesteps[i] # len(timesteps)=51
            t_next = timesteps[i + 1]

            # conditional
            x_cond = self.net(z, t.flatten(), labels)

            # unconditional
            x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))

            # cfg interval
            low, high = self.cfg_interval
            if low < 0: # power-cos cfg
                rescale_cfg_weight = (1 - torch.cos((t ** (-low)) * torch.pi)) * 1/2
                x_pred = x_cond + self.cfg_scale * rescale_cfg_weight.unsqueeze(1) * (x_cond - x_uncond)
            else:
                interval_mask = (t < high) & ((low == 0) | (t > low))
                cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
                x_pred = x_uncond + cfg_scale_interval.unsqueeze(1) * (x_cond - x_uncond)
            x_pred = x_pred / self.args.tau
            x_pred = x_pred.softmax(dim=1)
            B, classes, d, h, w = x_pred.shape
            x_pred = x_pred.permute(0,2,3,4,1) # [B,d,h,w,classes]
            pred_labels = torch.multinomial(x_pred.reshape(-1, classes), num_samples=1, replacement=True, generator=None).reshape(B,d,h,w)
            use_pred_mask = torch.rand(size=pred_labels.shape, device=pred_labels.device) < t_next
            z = torch.where(use_pred_mask, pred_labels, rand_labels)
        return z


    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
