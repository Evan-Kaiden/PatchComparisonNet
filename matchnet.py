import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18



class PatchEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        base = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.proj = nn.Linear(512, embed_dim)

    def forward(self, patches):
        feats = self.cnn(patches)
        feats = F.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        embeds = self.proj(feats)
        embeds = F.normalize(embeds, dim=1)
        return embeds
        
        
class Matcher(nn.Module):
    def __init__(self, num_classes, extractor, embed_dim=256, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = extractor
        self.encoder = PatchEncoder(embed_dim)
        self.temperature = temperature
        
    def encode_patches(self, x):
        patches = self.extractor(x)
        B, Tq, C, H, W = patches.shape
        patches = patches.reshape(B * Tq, C, H, W).contiguous()
        patch_embeds = self.encoder(patches)
        return patch_embeds, B, Tq

    def logits_from_batch_memory(self, x, y):

        patch_embeds, B, Tq = self.encode_patches(x)
        N = B * Tq

        y_patches = y.repeat_interleave(Tq)

        sim = patch_embeds @ patch_embeds.t() / self.temperature

        sim = sim - torch.eye(N, device=sim.device) * 1e9
        cls_one_hot = F.one_hot(y_patches, num_classes=self.num_classes).float()
        sim_per_class = sim @ cls_one_hot   

        logits = sim_per_class
        targets = y_patches

        return logits, targets

    def forward(self, x, y=None):
        if y is None:
            raise ValueError("For training this Matcher, call forward with x, y.")
        logits, targets = self.logits_from_batch_memory(x, y)
        return logits, targets

    
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

        sim = img_embeds @ mem_embeds.t() / self.temperature

        cls = cls.repeat_interleave(Tm)
        cls_one_hot = F.one_hot(cls, num_classes=self.num_classes).float()

        sim_per_class = sim @ cls_one_hot
        sim_per_class = sim_per_class.view(B, Tq, self.num_classes)

        logits = sim_per_class.sum(dim=1)
        return logits
