import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gumbel_topk_st


class PatchEncoder(nn.Module):
    def __init__(self, backbone, embed_dim=128):
        super().__init__()

        base = backbone
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.proj = nn.LazyLinear(embed_dim)

    def forward(self, patches):
        feats = self.cnn(patches)
        feats = F.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        embeds = self.proj(feats)
        # embeds = F.normalize(embeds, dim=1)
        return embeds
        
        
class Matcher(nn.Module):
    def __init__(self, num_classes, extractor, backbone, k=6, tau_gumbel=1.0, embed_dim=128, temperature=0.5):
        super().__init__()
        self.base_train = True
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

        for p in self.patch_scorer.parameters():
            p.requires_grad = False
            p.requires_grad = False

    def set_train_mode(self, base_train=True):
        self.base_train = base_train
        
    def scorer_train_mode(self):
        self.base_train = False

        for p in self.encoder.parameters():
            p.requires_grad = False

        for p in self.patch_scorer.parameters():
            p.requires_grad = True

        for m in self.patch_scorer.named_modules():
            if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
        
    def encode_patches(self, x):
        patches = self.extractor(x)
        B, Tq, C, H, W = patches.shape
        patches = patches.reshape(B * Tq, C, H, W).contiguous()
        patch_embeds = self.encoder(patches)
        return patch_embeds, B, Tq
    

    def forward(self, x, y, examples, examples_labels):
        query_patches = self.extractor(x)
        B, Tq, C, H, W = query_patches.shape
        query_patches = query_patches.reshape(B * Tq, C, H, W).contiguous()
        query_embeds = self.encoder(query_patches)
        
        ex_patches = self.extractor(examples)
        M, Tm, _, _, _ = ex_patches.shape
        ex_patches = ex_patches.reshape(M * Tm, C, H, W).contiguous()
        ex_embeds = self.encoder(ex_patches)
        
        sim = (query_embeds @ ex_embeds.t()) / self.temperature
        
        ex_labels_expanded = examples_labels.repeat_interleave(Tm)
        cls_one_hot = F.one_hot(ex_labels_expanded, num_classes=self.num_classes).float()
        sim_per_class = sim @ cls_one_hot
        sim_per_class = sim_per_class.view(B, Tq, self.num_classes)
        
        patch_targets = y.repeat_interleave(Tq)
        
        if self.base_train:
            logits = sim_per_class.sum(dim=1)
            patch_logits = sim_per_class
            return logits, patch_logits, patch_targets, torch.tensor(0, device=y.device)
        else:
            query_embeds_bt = query_embeds.view(B, Tq, -1)
            sel_logits = self.patch_scorer(query_embeds_bt).squeeze(-1)
            p = sel_logits.softmax(dim=-1)
            
            K = min(self.k, Tq)
            if self.training:
                w, idx = gumbel_topk_st(sel_logits, k=K, tau=self.tau_gumbel)
            else:
                idx = sel_logits.topk(K, dim=-1).indices
                w = torch.zeros_like(sel_logits).scatter_(1, idx, 1.0)
            
            logits = (sim_per_class * w.unsqueeze(-1)).sum(dim=1)
            patch_logits = sim_per_class
            return logits, patch_logits, patch_targets, p

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

        

        cls = cls.repeat_interleave(Tm)
        cls_one_hot = F.one_hot(cls, num_classes=self.num_classes).float()
        
        sim = (img_embeds @ mem_embeds.t()) / self.temperature
        sim_per_class = sim @ cls_one_hot
        sim_per_class = sim_per_class.view(B, Tq, self.num_classes)

        logits = (sim_per_class * w.unsqueeze(-1)).sum(dim=1)
        return logits