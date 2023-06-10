import torch
import models.resnets as resnets
import torch.nn as nn
import torch.distributions as td
from models.distributions import *
from utils import *
import os
import numpy as np

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
def load_baseVAE(datatype, configs, train_loader):
	if configs.model.act == "gelu": configs.model.activation = torch.nn.functional.gelu
	else: configs.model.act = nn.LeakyReLU()

	vae = VAE(configs, train_loader)
	return vae


def load_checkpoint(path, network):
	checkpoint = torch.load(path, map_location='cuda:0') 
	network.load_state_dict(checkpoint['model_state_dict'])
	return network

def save_checkpoint(path, network):
	torch.save({'model_state_dict': network.state_dict()}, path)

class VAE(nn.Module):
	def __init__(self, configs, data_loader=None):
		super().__init__()
        
		self.encoder =  resnets.FlatWideResNet(channels=configs.data.channels, size=configs.model.size, levels=configs.model.levels, blocks_per_level=configs.model.blocks_per_level, dense_blocks=configs.model.dense_blocks, activation = configs.model.activation, out_features = 2*configs.dist.latent_dim, shape=(configs.data.p,configs.data.q))
        
		self.decoder = resnets.FlatWideResNetUpscaling(channels=configs.data.channels, size=configs.model.size, levels=configs.model.levels, blocks_per_level=configs.model.blocks_per_level,  dense_blocks=configs.model.dense_blocks, activation = configs.model.activation, model = configs.model.model, transpose = configs.model.transpose, in_features = configs.dist.latent_dim, shape=(configs.data.p,configs.data.q)) #, skip=vaeac
		self.latent_dim = configs.dist.latent_dim
		self.configs = configs
		self.PATH = configs.model.path
		self.p_z = td.Independent(td.Normal(loc=torch.zeros(self.latent_dim).cuda(),scale=torch.ones(self.latent_dim).cuda()),1)
		self.encoder = self.encoder.cuda()
		self.decoder = self.decoder.cuda()
        
		if os.path.exists(configs.model.path+ 'trained_models/decoder.pt') :
			self.load_encoder(self.PATH+ 'trained_models/encoder.pt')
			self.load_decoder()
		else:
			self.train(data_loader)
		self.encoder.eval()
		self.decoder.eval()

		for params in self.encoder.parameters():
		    params.requires_grad = False

		for params in self.decoder.parameters():
		    params.requires_grad = False


	def load_encoder(self, path):
		self.encoder = load_checkpoint(path , self.encoder)
		
	def load_decoder(self):
		self.decoder = load_checkpoint(self.PATH + 'trained_models/decoder.pt', self.decoder)

	def start_train(self):
		self.encoder.train()
		for params in self.encoder.parameters():
			params.requires_grad = True

	def stop_train(self):
		self.encoder.eval()
		for params in self.encoder.parameters():
			params.requires_grad = False


	def encode(self, input):
		"""
		Encodes the input by passing through the encoder network
		and returns the latent codes. 
		:param input: (Tensor) input tensor to encode [N x C x H x W]
		:return: (Tensor) list of latent codes 
		"""

		out = self.encoder.forward(input)

		return out 

	def decode(self, z):
		"""
		Decodes samples from the latent distribution z through the decoder network 
		and returns the parameters of the data distribution 
		:param z: (Tensor) input tensor to decode [N x self.latent_dim]
		:return: (Tensor) tensor of parameters for data distribution [N x C x H x W] 
		"""
		return self.decoder.forward(z)

	def forward(self, input, K=1):
		params = self.encode(input)
		z, q_z = self.posterior_sample(params, K)
		return [self.decode(z), params]

	def data_dist(self, out, form='gaussian'):
		"""
		Initializes a distribution.
		:param form: the form of the distribution, either Gaussian or point estimate (Dirac delta).
		:return: the initialized distribution
		"""
		if form == 'DiscNormal':
			sigma_decoder = self.decoder.get_parameter("log_sigma").cuda()
			return DiscNormal(loc = out.reshape([-1,1]), scale = (torch.nn.Softplus()(sigma_decoder))*(torch.ones(*out.shape, device=device)).reshape([-1,1]))
		elif form == 'ContBernoulli':
			return continuous_bernoulli(out)
		else:
			raise Exception('Distribution form not found.')
            
            
	def output_dist_mean(self, x):
		dist = self.data_dist(x, self.configs.dist.data)
		return dist, dist.mean()

	def impute(self, b_data, b_mask, params, K, batches=100):
		if len(b_data.shape) == 2: [batch_size, p] = b_data.shape
		else: [batch_size, channels, p, q] = b_data.shape

		batch_size = b_data.shape[0]
		dims = int(np.prod(b_data.shape)/batch_size)
		n_batches = int((K*batch_size)/batches) 
		samples_per_batch = int(batches/batch_size) 

		data_tiled = data_replicate(b_data, samples_per_batch) 
		mask = data_replicate(b_mask, samples_per_batch) 
		imputation_miss = data_replicate(b_data, K) 

		for i in range(n_batches):
			z, q_z = self.posterior_sample(params, samples_per_batch)
			z = z.reshape([samples_per_batch*batch_size, self.latent_dim])
			decoder_out = self.decode(z)
			imputations = data_tiled
			imputations[~mask] = self.output_dist_mean(decoder_out)[1].reshape([samples_per_batch*batch_size,channels,p,q])[~mask]
			imputation_miss[i*batches: (i+1)*batches] = imputations[:]

		return imputation_miss.reshape(K,batch_size, channels,p,q)
          
	def get_log_qz(self,q_z, z):
		return q_z.log_prob(z)
    
	def iwae(self, b_data, b_mask, b_full, params, K, batches=1000):
		[batch_size, channels, p, q] = b_data.shape
		n_batches = int((K*batch_size)/batches)
		samples_per_batch = int(batches/batch_size) 

		logpxmissgivenz = torch.zeros((K*batch_size,1)).cuda()
		logpz = torch.zeros((K*batch_size,1)).cuda()
		logqz = torch.zeros((K*batch_size,1)).cuda()
		mse_samples = np.zeros((K*batch_size))
        
		full_ = data_replicate(b_full, samples_per_batch) 
		mask =  data_replicate(b_mask, samples_per_batch)
		data_tiled = data_replicate(b_data, samples_per_batch) 

		p_z = td.Independent(td.Normal(loc=torch.zeros(self.latent_dim).cuda(),scale=torch.ones(self.latent_dim).cuda()),1)
		data_flat = full_.reshape([-1,1])

		for i in range(n_batches):
			z, q_z = self.posterior_sample(params, samples_per_batch)
			logqz_batch = self.get_log_qz(q_z, z).reshape(batches,1)
			logqz[i*batches: (i+1)*batches] = logqz_batch
			logpz[i*batches: (i+1)*batches] = p_z.log_prob(z).reshape(batches,1) 

			z_flat = z.reshape([samples_per_batch*batch_size, self.latent_dim])
			decoder_out = self.decode(z_flat)
			imputations = data_tiled
			dist, output_mean = self.output_dist_mean(decoder_out)
			output_mean = output_mean.reshape([samples_per_batch*batch_size,channels,p,q])
			imputations[~mask] = output_mean[~mask]
			#imputation_miss[i*batches: (i+1)*batches][~mask] = imputations[:]

			##q_z might need fixing for flow dist
			mse_per_pixel = mse_(imputations.cpu().data.numpy().reshape([samples_per_batch*batch_size,channels*p*q]),full_.cpu().data.numpy().reshape([samples_per_batch*batch_size,channels*p*q]))

			all_log_pxgivenz = dist.log_prob(data_flat).reshape([samples_per_batch*batch_size,channels*p*q])
			logmiss = torch.sum(all_log_pxgivenz*(~mask.reshape([samples_per_batch*batch_size,channels*p*q])),1).reshape([samples_per_batch*batch_size,1]) 			
			logpxmissgivenz[i*batches: (i+1)*batches] = logmiss
            
			mse_samples[i*batches: (i+1)*batches] = mse_per_pixel
            
		mse_samples = mse_samples.reshape(K,batch_size)
		logpxmissgivenz = logpxmissgivenz.reshape(K, batch_size)
		logpz = logpz.reshape(K, batch_size)
		logqz = logqz.reshape(K, batch_size)
		iwae = np.zeros((K,batch_size))

		for i in np.arange(K):
			iwae[i,:] = torch.logsumexp(logpxmissgivenz[:i,:] + logpz[:i,:] - logqz[:i,:], 0).cpu().data - np.log(i+1) 

		return iwae, mse_samples

	def get_params(self, b_data, b_mask = None, i=None):
		return self.encode(b_data)

	def evaluate_batch(self, batch_data, params):
		n_mini_batches = int(self.configs.qavi.batches/self.configs.qavi.K)
		[b_data, b_mask, b_full] = batch_data
		if len(b_data.shape) ==2: [batch_size, p] = b_data.shape
		else:  [batch_size, channels, p, q] = b_data.shape
		iwae_batch = np.zeros((10000,batch_size))
		mse_batch = np.zeros((10000,batch_size))
		imputation_batch = data_replicate(b_data, 100).reshape([100, batch_size, channels, p, q])

		for j in range(int(batch_size/n_mini_batches)):
			mini_b_data = jth_batch(b_data, j, n_mini_batches)
			mini_b_mask = jth_batch(b_mask, j, n_mini_batches)
			mini_b_full = jth_batch(b_full, j, n_mini_batches)
			iwae, mse = self.iwae(mini_b_data, mini_b_mask, mini_b_full, params[j], K=10000, batches=1000) # K , batch_size
			imputations = self.impute(mini_b_data, mini_b_mask, params[j], K=100, batches=100) # K, batch_size, channels, p, q
			##Put the iwae, mse, imputations for the batch
			iwae_batch[:,j*n_mini_batches: (j+1)*n_mini_batches] = iwae
			mse_batch[:,j*n_mini_batches: (j+1)*n_mini_batches] = mse
			imputation_batch[:,j*n_mini_batches: (j+1)*n_mini_batches] = imputations

		return [iwae_batch, mse_batch, imputation_batch]

	def evaluate(self, data_loader, m_type=None):
		num_batches = len(data_loader)
		if not os.path.exists(self.file_name+m_type + "/parameters.pkl"):
			print("parameters aren't saved.... can't proceed for evaluations... returning")
			return
		read_only = False

		if os.path.exists(self.file_name + m_type + "/evaluations.pkl") and check_file(self.file_name + m_type + "/evaluations.pkl", num_batches): 
			print("evaluations are available, reading file instead of computing") 
			read_only = True
			stats_full = read_file(self.file_name + m_type + "/evaluations.pkl", num_batches)  
		else: print("Computing IWAE")

		mean_iwae = 0
		for i, data in enumerate(data_loader): 
			b_data, b_mask, b_full = data[0:3]
			b_data = b_data.to(device,dtype = torch.float)
			b_mask = b_mask.to(device,dtype = torch.bool)
			b_full = b_full.to(device,dtype = torch.float)
            
			if len(b_data.shape) ==2: [batch_size, p] = b_data.shape
			else:  [batch_size, channels, p, q] = b_data.shape   
            
			params = self.get_params(b_data, b_mask, i, m_type)
			#stats is [iwae_batch, mse_batch, imputation_batch]
			if read_only: 
				stats = stats_full[i]
			else: 
				stats = self.evaluate_batch([b_data, b_mask, b_full], params) 
				s_file(self.file_name + m_type + "/evaluations.pkl", stats, 'ab+') 
             
			n_mask = torch.sum(~b_mask, (1,2,3)).cpu().data.numpy()/channels   
			batch_iwae = stats[0]
			mean_iwae += np.mean(np.divide(batch_iwae[-1,:],n_mask))
            
		print("IWAE: ", mean_iwae/num_batches)            
		return

	def posterior_sample(self, params, K=1):
		"""
		Samples one z sample in the latent space per datapoint
		:param mu: (Tensor) mean of the latent Gaussian
		:param log_std: (Tensor) Standard deviation of the latent Gaussian
		:return: a sample from the posterior
		"""
		mu = params[..., :self.latent_dim]
		log_std = params[..., self.latent_dim:] 
		q_z = td.Independent(td.Normal(loc=mu,scale= self.configs.dist.min_std + torch.nn.Softplus()(log_std) ),1)
		z = q_z.rsample([K])
		return z, q_z

	def recon_prob(self, out_decoder, input_):
		out_dist = self.data_dist(out_decoder, self.configs.dist.data) 
		all_log_pxgivenz = out_dist.log_prob(input_)
		return all_log_pxgivenz


	def latent_loss(self, q_z=None, z=None, params=None):
		mean = params[..., :self.latent_dim] 
		log_std = params[..., self.latent_dim:] 
		stdev = self.configs.dist.min_std + torch.nn.Softplus()(log_std)
		mean_sq = mean * mean
		stddev_sq = stdev * stdev
		return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1,1)

    
	def loss(self, input_):
		[batch_size, channels, p, q] = input_.shape
		[recon, params]  = self.forward(input_)

		input_flat = input_.reshape([-1, 1])
		recon_prob = self.recon_prob(recon,input_flat).reshape([batch_size,channels*p*q])
		recon_prob_batch = torch.sum(recon_prob,[1]).reshape(batch_size)

		kld = self.latent_loss(params = params).reshape(batch_size) 
		loss = -torch.mean(recon_prob_batch - kld)
		print(loss, recon_prob_batch, kld)
		return {"loss" : loss, "recon_prob": torch.mean(recon_prob_batch), "kl" : torch.mean(kld)}

	def train(self, data_loader):
		optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.configs.optim.lr)
		best_loss = 10000
		elbo_trace, kl_trace, log_like_trace = [], [], []
		for epoch in range(self.configs.optim.epochs):
			print(epoch)
			train_loss, nb, kl_epoch, recon_prob_epoch = 0, 0, 0, 0

			nb = 0
			for data in data_loader: 
				b_data = data[2]
				loss_dict = self.loss(b_data.to(device,dtype = torch.float)) #.cuda().float()
				loss_dict['loss'].backward()
				optimizer.step()

				train_loss += float(loss_dict['loss'])
				kl_epoch += float(loss_dict['kl'])
				recon_prob_epoch += float(loss_dict['recon_prob'])
				nb += 1

			elbo_trace.append(-train_loss/nb)
			kl_trace.append(kl_epoch/nb)
			log_like_trace.append(log_like_epoch/nb)

			print('Epoch {}: Training : Loss {}'.format(epoch, train_loss))

			torch.save({'model_state_dict': self.decoder.state_dict()}, self.PATH + 'trained_models/decoder.pt')
			torch.save({'model_state_dict': self.encoder.state_dict()}, self.PATH + 'trained_models/encoder.pt')

			s_file(self.PATH + 'trained_models/traces.pkl',dict(elbo=elbo_trace, kl=kl_trace, log_like=log_like_trace) )










 


