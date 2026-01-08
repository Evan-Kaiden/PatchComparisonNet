import torch
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/pdf/2004.11362v1 (Supervised Contrastive Learning)



class SupervisedContrastiveLoss(nn.Module):
   def __init__(self, temperature=0.1):
       super().__init__()
       self.temperature = temperature


   def forward(self, feats, labels):
       device = feats.device
       N = feats.size(0)

       labels = labels.contiguous().view(-1, 1)
       mask = torch.eq(labels, labels.T).float().to(device)


       # Use negative squared Euclidean distance as similarity
       # sim[i,j] = -||feats[i] - feats[j]||^2
       feats_square = (feats ** 2).sum(dim=1, keepdim=True)
       dist_square = feats_square + feats_square.T - 2 * (feats @ feats.T)
       logits = -dist_square / self.temperature


       logits_max, _ = torch.max(logits, dim=1, keepdim=True)
       logits = logits - logits_max.detach()


       logits_mask = torch.ones_like(mask) - torch.eye(N, device=device)
       mask = mask * logits_mask


       exp_logits = torch.exp(logits) * logits_mask
       log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)


       pos_counts = mask.sum(dim=1)
       valid = pos_counts > 0
      
       if valid.sum() == 0:
           return torch.tensor(0.0, device=device, requires_grad=True)
      
       valid_f = valid.float()
       numerator = (mask * log_prob).sum(dim=1)
       denominator = pos_counts.clamp_min(1)
       mean_log_prob_pos = numerator / denominator
       mean_log_prob_pos = mean_log_prob_pos * valid_f
       loss = -(mean_log_prob_pos * valid_f).sum() / valid_f.sum().clamp_min(1)
      
       return loss


    
def gumbel_topk_st(logits: torch.Tensor, k: int, tau: float):
    g = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + g
    idx = y.topk(k, dim=-1).indices

    w_hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)
    w_soft = F.softmax(logits / tau, dim=-1)
    weights = w_hard + (w_soft - w_soft.detach())
    return weights, idx

def get_scheduler(map_arg, optimizer, scheduler, epochs, lr):
    if scheduler == "cosine":
        scheduler = map_arg[scheduler](optimizer=optimizer, T_max=epochs, eta_min=(lr / 10))
    elif scheduler == "linear":
        scheduler = map_arg[scheduler](optimizer=optimizer, total_iters=epochs, start_factor=1, end_factor=.75)
    elif scheduler == "step":
        scheduler = map_arg[scheduler](optimizer=optimizer, step_size=max(1, epochs // 10), gamma=0.5)
    else:
        scheduler = None
    
    return scheduler