import torch
import torch.distributions as td
from torch.distributions import Normal
import numpy as np


def inject_grads(final_samples, logits, means, scales):
    final_samples = final_samples.detach() 

    # Reparameterize logits
    logit_dist = td.MixtureSameFamily(td.Categorical(logits=logits), td.Independent(td.Normal(means.detach(), scales.detach()), 1))
    logit_prob = logit_dist.log_prob(final_samples)
    logit_injection = torch.exp(logit_prob - logit_prob.detach()).unsqueeze(-1)

    # Reparameterize means and scales
    logits = logits.detach()
    # B, 3, H, W
    samples = final_samples.unsqueeze(-2).expand(*((final_samples.ndim - 1) * [-1] + [logits.shape[-1], -1]))
    log_probs = td.Normal(means, scales).log_prob(samples)
    Fcdf = td.Normal(means, scales).cdf(samples)
    
    rsamples = []
    for j in range(samples.shape[-1]):
        g = (Fcdf[..., j] * torch.exp(logits)).sum(-1)
        logits = log_probs[..., j] + logits # B, H,  M
        q = torch.exp(logits.logsumexp(-1)).detach() # B, H
        grad = - g / q    # B, H
        sampleij = final_samples[..., j] + (grad - grad.detach())   # B, H
        logits = (logits - logits.logsumexp(-1, True)) # B, H, M
        rsamples.append(sampleij)
        
    rsample = torch.stack(rsamples, dim=-1) # B, 3, H    
    return logit_injection * rsample

class ReparameterizedNormalMixture1d(td.MixtureSameFamily):
    '''
    Means/Scales should be of shape: [Batch size, # components, # dimensions]
    Logits should be of shape: [Batch size, # components]
    '''
    def __init__(self, logits, means, scales):
        assert means.ndim - 1 == logits.ndim
        assert scales.ndim - 1 == logits.ndim
        super(ReparameterizedNormalMixture1d, self).__init__(td.Categorical(logits=logits), td.Independent(td.Normal(means, scales), 1))
        self.logits = logits
        self.means = means
        self.scales = scales

    def rsample(self, sample_shape=torch.Size(), beta=1):
        sample = self.sample(sample_shape)
        return inject_grads(sample, self.logits, self.means, self.scales)



