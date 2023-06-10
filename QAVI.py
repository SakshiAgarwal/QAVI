import torch
from models.vae import *
from pyro.nn import AutoRegressiveNN
import pyro
from pyro.distributions.transforms import AffineAutoregressive
from utils import *
from datetime import datetime
import numpy as np
import pickle
from mixture import *

'''
This script contains implementations on Gaus, Flow and Mix. QAVI
'''
class Gaus_QAVI(VAE):
	def __init__(self, configs, test_loader=None):
		super().__init__(configs)
		self.file_name = configs.model.path + 'results/qavi/gaus/'
		self.params = [{} for i in range(1000)] 

	def b_init_params(self,b_data):
		params  = self.encoder.forward(b_data)
		params[...,self.latent_dim:] = torch.ones(b_data.shape[0], self.latent_dim)
		params = params.cuda()
		return params

	def init_params(self, b_data):
		params = self.b_init_params(b_data)
		return params
                
	def get_params(self, b_data, b_mask, i, m_type=None):
		n_mini_batches = int(self.configs.qavi.batches/self.configs.qavi.K)
		params = []
		with open(self.file_name + m_type + "/parameters.pkl", 'rb') as file:
			j = 0
			while j<i: 
				for k in range(n_mini_batches) : params_batch = pickle.load(file)
				j += 1
			for j in range(n_mini_batches) : params.append(pickle.load(file))  
		return params           

	def loss(self, b_data, b_mask, params, K=10):
		batch_size = b_data.shape[0]
		n_images = int(batch_size/K)

		dims = int(np.prod(b_data.shape)/batch_size)

		z, q_z = self.posterior_sample(params, K = K)
		recon = self.decode(z)

		input_flat = b_data.reshape([-1, 1])
		b_mask = b_mask.reshape([batch_size, dims])
		recon_prob = self.recon_prob(recon,input_flat).reshape([batch_size,dims])
		recon_prob_batch = torch.sum(recon_prob*b_mask,[1]).reshape(K,n_images)

		kld = self.latent_loss(q_z, z, params).reshape(n_images) 
		loss = -torch.mean(torch.mean(recon_prob_batch,0) - self.configs.qavi.beta*kld)
		nelbo = -torch.mean(torch.mean(recon_prob_batch,0) - kld)
		#print(loss)
		return {"loss" : loss, "recon_prob": torch.mean(recon_prob_batch), "kl" : torch.mean(kld), "nelbo": nelbo}

	def params_train(self, params):
		params.requires_grad = True
		return params

	def params_eval(self, params):
		params.requires_grad = False
		return params
    
	def get_optimizers(self, params) : 
		[lr, step_size, gamma] = [1.0, 30, 0.5]
		optimizer = torch.optim.Adam([params], lr=lr, betas=(0.9, 0.999))
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
		return [optimizer], [scheduler]

	def phase_two(self, params, optimizer, scheduler, k):
		return params, optimizer, scheduler

	def phase_one(self, params, optimizer, scheduler, k):
		return params, optimizer, scheduler

	def re_init(self, b_data, params, optimizers):
		return params, optimizers

	def train_mini_batch(self, iterations, K, data_args): 
		[data, mask, full] = data_args
		params = self.init_params(data)
		data_ = data_replicate(data, K)
		mask_ = data_replicate(mask, K)
		full_ = data_replicate(full, K)

		params = self.params_train(params)
		optimizers, schedulers = self.get_optimizers(params)
		loss_batch, nelbo_batch, time_batch = np.zeros(iterations), np.zeros(iterations),np.zeros(iterations)
		k1 = 50 
		start = datetime.now()
		for k in range(iterations):
			if k == k1 : params, optimizers, schedulers = self.phase_one(params, optimizers, schedulers, k)
			if k == int(2*iterations/3): params, optimizers, schedulers = self.phase_two(params, optimizers, schedulers, k)
			loss_ = self.loss(data_, mask_, params, K)
			for opt in optimizers: opt.zero_grad()
			loss_["loss"].backward()
			for opt in optimizers: opt.step()
			for sch in schedulers: sch.step()
            
			if k%int((k1+1)/5)==0 and k<=k1-1: params, optimizers = self.re_init(data, params, optimizers)
			loss_batch[k] += loss_["loss"].item()
			nelbo_batch[k] += loss_["nelbo"].item()
			end = datetime.now()    
			diff = end-start 
			time_batch[k] += diff.total_seconds()
			if k%50==0: print(k, loss_["loss"].item())

            
		params = self.params_eval(params)
		return params, [loss_batch, nelbo_batch, time_batch]        
   
	def train(self, iterations, K, data, m_type=None): 
		batches = self.configs.qavi.batches
		[b_data, b_mask, b_full] = data
		batch_size = b_data.shape[0]
		n_batches = int((K*batch_size)/batches)
		loss_trace, nelbo_trace, time_trace = [], [], []
		loss_batch, nelbo_batch, time_batch = np.zeros(iterations), np.zeros(iterations),np.zeros(iterations)
		params = []        
		for j in range(n_batches):
			print(j)
			mini_b_data = jth_batch(b_data, j, batches/K)
			mini_b_mask = jth_batch(b_mask, j, batches/K)
			mini_b_full = jth_batch(b_full, j, batches/K)
			params_batch, stats = self.train_mini_batch(iterations, K, [mini_b_data, mini_b_mask, mini_b_full])       
            
			if m_type is not None: s_file(self.file_name + m_type + "/parameters.pkl", params_batch, "ab+") 
            
			loss_batch +=  stats[0]        
			nelbo_batch +=  stats[1]              
			time_batch +=  stats[2]  
			params.append(params_batch)
            
		loss_trace.append(loss_batch/j)
		nelbo_trace.append(nelbo_batch/j)
		time_trace.append(time_batch/j)

		if m_type is not None: s_file(self.file_name + m_type + "/traces.pkl", [loss_trace, nelbo_trace, time_trace], 'ab+') 
		return params
   
	def show_imputations(self, data_loader, m_type=None):
		num_batches = len(data_loader)
		read_only = False

		if not os.path.exists(self.file_name + m_type):
			os.mkdir(self.file_name + m_type)     
            
		if os.path.exists(self.file_name + m_type + "/parameters.pkl") and check_file(self.file_name + m_type + "/parameters.pkl", num_batches*int(self.configs.qavi.batches/self.configs.qavi.K)): 
			print("parameters available, reading file instead of training") 
			read_only = True
            
		for i, data in enumerate(data_loader): 
			b_data, b_mask, b_full = data[0:3]
			b_data = b_data.to(device,dtype = torch.float)
			b_mask = b_mask.to(device,dtype = torch.bool)
			b_full = b_full.to(device,dtype = torch.float)

			if not read_only: params = self.train(self.configs.qavi.iterations, self.configs.qavi.K, [b_data, b_mask, b_full], m_type) 
			else: params =  self.get_params(b_data, b_mask, i, m_type)
            
			if i == (num_batches - 1):
				print("Displaying imputations for 10 test data points")
				mini_b_data = jth_batch(b_data, 0, int(self.configs.qavi.batches/self.configs.qavi.K))
				mini_b_mask = jth_batch(b_mask, 0, int(self.configs.qavi.batches/self.configs.qavi.K))
				mini_b_full = jth_batch(b_full, 0, int(self.configs.qavi.batches/self.configs.qavi.K))
				imputations = self.impute(mini_b_data, mini_b_mask, params[0], K=10, batches=100) # K, batch_size, channels, p, q
				show_images(mini_b_data, mini_b_full, imputations)
    

class Flow_QAVI(Gaus_QAVI):
	def __init__(self, configs, test_loader=None):
		super().__init__(configs)
		self.file_name = configs.model.path + 'results/qavi/flow/'

	def init_params(self, mini_b_data):
		base_params = self.b_init_params(mini_b_data)
		batch_size = mini_b_data.shape[0]
		autoregressive_nn =  AutoRegressiveNN(self.latent_dim, [320, 320]).cuda()
		autoregressive_nn2 = AutoRegressiveNN(self.latent_dim, [320, 320]).cuda() 

		sd = autoregressive_nn.state_dict()
		sd['layers.2.weight'] = sd['layers.2.weight']*1e-4
		sd['layers.2.bias'] = sd['layers.2.bias']*1e-4
		autoregressive_nn.load_state_dict(sd)

		sd = autoregressive_nn2.state_dict()
		sd['layers.2.weight'] = sd['layers.2.weight']*1e-4
		sd['layers.2.bias'] = sd['layers.2.bias']*1e-4
		autoregressive_nn2.load_state_dict(sd)

		return {"base_params" : base_params, "L1" : autoregressive_nn, "L2" : autoregressive_nn2}

	def params_train(self, params): 
		params["base_params"].requires_grad = True
		for params_ in params["L1"].parameters():
			params_.requires_grad = True
		for params_ in params["L2"].parameters():
			params_.requires_grad = True
		return params

	def params_eval(self, params):
		params["base_params"].requires_grad = False
		for params_ in params["L1"].parameters():
			params_.requires_grad = False
		for params_ in params["L2"].parameters():
			params_.requires_grad = False
		return params

	def get_optimizers(self, params):
		lr = 0.01
		lr_b = 0.1
		optimizer = torch.optim.Adam([params["base_params"]], lr=lr_b, betas=(0.9, 0.999))  
		total_params = list()
		total_params.extend(params["L1"].parameters())
		total_params.extend(params["L2"].parameters())
		optimizer_iaf = torch.optim.Adam(total_params, lr=lr)
		return [optimizer, optimizer_iaf], []

	def posterior_sample(self, params, K):
		mu = params["base_params"][..., :self.latent_dim]
		log_std = params["base_params"][..., self.latent_dim:] 
		q_z = td.Independent(td.Normal(loc=mu,scale= self.configs.dist.min_std + torch.nn.Softplus()(log_std) ),1)
		transforms = []
		transforms.append(AffineAutoregressive(params["L1"]).cuda())
		transforms.append(AffineAutoregressive(params["L2"]).cuda())
		flow_dist = pyro.distributions.torch.TransformedDistribution(q_z, transforms)
		z_samples = flow_dist.rsample([K])
		return z_samples, flow_dist
    
	def latent_loss(self, q_z, z, params):
		return approx_latent_loss(self.p_z, q_z, z, params)    
    
class Mix_QAVI(Gaus_QAVI):
	def __init__(self, configs, test_loader=None):
		super().__init__(configs)
		self.num_components = 10
		self.file_name = configs.model.path + 'results/qavi/mix/'

	def b_init_params(self, b_data, r1=-1, r2=1):
		batch_size = b_data.shape[0]
		out_encoder = self.encoder.forward(b_data) #.reshape((batch_size, num_components, 2*d))
		out_encoder = data_replicate(out_encoder, self.num_components).reshape((batch_size, self.num_components, 2*self.latent_dim))

		logits = torch.zeros(batch_size, self.num_components).cuda()
		means = torch.zeros(batch_size, self.num_components, self.latent_dim).cuda()
		log_std = torch.ones(batch_size, self.num_components, self.latent_dim).cuda()

		means[:,...] = out_encoder[...,:self.latent_dim]
		means += (r2-r1) * torch.rand(batch_size, self.num_components, self.latent_dim).cuda() + r1 
		return {"logits" : logits, "means": means, "log_std" : log_std}


	def get_optimizers(self, params) : 
		optimizer = torch.optim.Adam([params["means"],params["log_std"]], lr=1.0, betas=(0.9, 0.999))
		optimizer_logits = torch.optim.Adam([params["logits"]], lr=0.1, betas=(0.9, 0.999)) 

		return [optimizer, optimizer_logits], [] 

	def phase_one(self, params, optimizer, scheduler, k):
		[lr, step_size, gamma] = [1.0, 30, 0.5]
		params["logits"] = torch.zeros(params["logits"].shape[0], self.num_components).cuda()
		params["logits"].requires_grad = False
		optimizer = torch.optim.Adam([params["means"],params["log_std"]], lr=lr, betas=(0.9, 0.999)) #was 0.1
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
		return params, [optimizer], [scheduler]
    
	def phase_two(self, params, optimizer, scheduler, k):
		[lr_logits, ss_logits, g_logits] = [0.1, 30, 0.1]
		params['logits'].requires_grad = True
		optimizer_logits = torch.optim.Adam([params["logits"]], lr=lr_logits, betas=(0.9, 0.999)) 
		scheduler_logits = torch.optim.lr_scheduler.StepLR(optimizer_logits, step_size=ss_logits, gamma=g_logits)
		optimizer.append(optimizer_logits)
		scheduler.append(scheduler_logits)
		return params, optimizer, scheduler

	def re_init(self, b_data, params, optimizers):
		threshold = 0.07 
		probs = torch.softmax(params["logits"].detach(),dim=1)
		ap = torch.any(probs<threshold)
		if ap:
			params_re_init = self.b_init_params(b_data)
			## Weigh every component equally
			params["logits"] = params_re_init["logits"]

			##Update means, log_std for components weights < threshold
			means_curr = params["means"].detach()
			scales_curr = params["log_std"].detach()
			probs = torch.Tensor.repeat(probs.reshape(probs.shape[0], self.num_components, 1),[1,1,self.latent_dim]) 
			params["means"] = torch.where(probs >= threshold, means_curr, params_re_init["means"])
			params["log_std"] = torch.where(probs >= threshold, scales_curr, params_re_init["log_std"])
			params = params_train(params)
			
			lr_ = optimizers[0].param_groups[0]["lr"]
			optimizers[0] = torch.optim.Adam([params["means"],params["log_std"]], lr=lr_, betas=(0.9, 0.999)) 
			optimizers[1] = torch.optim.Adam([params["logits"]], lr=0.1, betas=(0.9, 0.999)) 
		return params, optimizers


	def posterior_sample(self, params, K):
		mu = params["means"]
		log_std = params["log_std"]

		q_z = ReparameterizedNormalMixture1d(params["logits"], params["means"], self.configs.dist.min_std + torch.nn.Softplus()(params["log_std"]) )
		zgivenx = q_z.rsample([K])
		return zgivenx, q_z

	def latent_loss(self, q_z, z, params):
		return approx_latent_loss(self.p_z, q_z, z, params)

	def params_train(self, params): 
		params["logits"] = params["logits"].to(device,dtype = torch.float)
		params["means"] = params["means"].to(device,dtype = torch.float)
		params["log_std"]= params["log_std"].to(device,dtype = torch.float)

		params["logits"].requires_grad = False
		params["means"].requires_grad = True
		params["log_std"].requires_grad = True
		return params

	def params_eval(self, params):
		params["logits"].requires_grad = False
		params["means"].requires_grad = False
		params["log_std"].requires_grad = False
		return params
