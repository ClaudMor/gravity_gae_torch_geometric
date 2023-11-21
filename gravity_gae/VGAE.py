
import torch
import copy
from torch.nn import Module

class VGAE_Reparametrization(Module):
    def __init__(self, MAX_LOGSTD = None, num_noise_samples = 100):
        super().__init__()

        self.MAX_LOGSTD = MAX_LOGSTD
        self.num_noise_samples = num_noise_samples

    def forward(self, mu_logstd):
        new_batch = copy.copy(mu_logstd[0])

        mu_batch, logstd_batch = mu_logstd

        
        mu, logstd = mu_batch.x, logstd_batch.x

        if self.MAX_LOGSTD is not None:
            logstd = logstd.clamp(max = self.MAX_LOGSTD)


        if self.training:
            new_batch.x =  mu + torch.randn_like(logstd) *  torch.exp(logstd)
        else:
            new_batch.x = mu

        new_batch.mu = mu
        new_batch.logstd = logstd

        return new_batch