import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gumbel_topk_st


class PatchEncoder(nn.Module):
    def __init__(self, backbone, embed_dim=256):
        super().__init__()

        base = backbone
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.proj = nn.LazyLinear(embed_dim)

    def forward(self, patches):
        feats = self.cnn(patches)
        feats = F.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        embeds = self.proj(feats)
        embeds = F.normalize(embeds, dim=1)
        return embeds
        
        
class Matcher(nn.Module):
    def __init__(self, num_classes, extractor, backbone, k=8, tau_gumbel=1.0, embed_dim=256, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = extractor
        self.encoder = PatchEncoder(backbone, embed_dim)
        self.temperature = temperature

        self.k = k
        self.tau_gumbel = tau_gumbel
        self.patch_scorer = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        for m in self.patch_scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)
        
    def encode_patches(self, x):
        patches = self.extractor(x)
        B, Tq, C, H, W = patches.shape
        patches = patches.reshape(B * Tq, C, H, W).contiguous()
        patch_embeds = self.encoder(patches)
        return patch_embeds, B, Tq

    def logits_from_batch_memory(self, x, y):
        patch_embeds, B, Tq = self.encode_patches(x)
        N = B * Tq

        patch_embeds_bt = patch_embeds.view(B, Tq, -1)
        sel_logits = self.patch_scorer(patch_embeds_bt).squeeze(-1)
        p = sel_logits.softmax(dim=-1)

        K = min(self.k, Tq)
        if self.training:
            w, idx = gumbel_topk_st(sel_logits, k=K, tau=self.tau_gumbel)
        else:
            idx = sel_logits.topk(K, dim=-1).indices
            w = torch.zeros_like(sel_logits).scatter_(1, idx, 1.0)

        y_patches = y.repeat_interleave(Tq)

        sim = (patch_embeds @ patch_embeds.t()) / self.temperature
        sim = sim - torch.eye(N, device=sim.device) * 1e9

        cls_one_hot = F.one_hot(y_patches, num_classes=self.num_classes).float()
        sim_per_class = sim @ cls_one_hot

        w_flat = w.reshape(N)
        logits = sim_per_class * w_flat.unsqueeze(1)

        targets = y_patches
        return logits, targets, p

    def forward(self, x, y=None):
        if y is None:
            raise ValueError("For training this Matcher, call forward with x, y.")
        return self.logits_from_batch_memory(x, y)

    @torch.no_grad()
    def predict(self, x, memory, cls):
        img_patches = self.extractor(x)
        mem_patches = self.extractor(memory)

        B, Tq, C, H, W = img_patches.shape
        M, Tm, _, _, _ = mem_patches.shape

        img_patches = img_patches.reshape(B * Tq, C, H, W).contiguous()
        mem_patches = mem_patches.reshape(M * Tm, C, H, W).contiguous()

        img_embeds = self.encoder(img_patches)
        mem_embeds = self.encoder(mem_patches)

        img_embeds_bt = img_embeds.view(B, Tq, -1)
        sel_logits = self.patch_scorer(img_embeds_bt).squeeze(-1)

        K = min(self.k, Tq)
        idx = sel_logits.topk(K, dim=-1).indices
        w = torch.zeros_like(sel_logits).scatter_(1, idx, 1.0)

        sim = (img_embeds @ mem_embeds.t()) / self.temperature

        cls = cls.repeat_interleave(Tm)
        cls_one_hot = F.one_hot(cls, num_classes=self.num_classes).float()

        sim_per_class = sim @ cls_one_hot
        sim_per_class = sim_per_class.view(B, Tq, self.num_classes)

        logits = (sim_per_class * w.unsqueeze(-1)).sum(dim=1)
        return logits