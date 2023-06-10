import torch.distributions as td
import torch

class continuous_bernoulli():
	def __init__(self,out):
		self.outs = out
		self.dist = td.continuous_bernoulli.ContinuousBernoulli(logits=out.reshape([-1,1]))

	def log_prob(self,x):
		return self.dist.log_prob(x)

	def mean(self):
		return torch.sigmoid(self.outs)

