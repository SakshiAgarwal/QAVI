import ml_collections

def mnist_configs():
	config = ml_collections.ConfigDict()

	# architecture of encoder and decoder
	config.model = model = ml_collections.ConfigDict()
	model.size = 1
	model.act = "gelu"
	model.blocks_per_level = 2
	model.levels = 3
	model.dense_blocks = 0
	model.model = "vae"
	model.transpose = False
	model.path = "mnist/"
    
	# data
	config.data = data = ml_collections.ConfigDict()
	data.p , data.q = 28, 28
	data.channels = 1 
	data.dataset = "MNIST"

	# distributions 
	config.dist = dist = ml_collections.ConfigDict()
	dist.latent = "Gaussian"
	dist.latent_dim = 50
	dist.min_std = 0
	dist.data = "ContBernoulli"

	#base VAE optimization
	config.optim = optim = ml_collections.ConfigDict()
	optim.lr = 1e-3
	optim.optimizer = "Adam"
	optim.epochs = 1000

	#QAVI hyperparameters
	config.qavi = qavi = ml_collections.ConfigDict()
	qavi.beta = 20
	qavi.batches = 1000
	qavi.K = 100
	qavi.iterations = 300
	return config


def svhn_configs():
	config = ml_collections.ConfigDict()

	# architecture of encoder and decoder
	config.model = model = ml_collections.ConfigDict()
	model.size = 2
	model.act = "relu"
	model.blocks_per_level = 4
	model.levels = 3
	model.dense_blocks = 2
	model.model = "sigma_vae"
	model.transpose = True
	model.path = "svhn/"

	# data
	config.data = data = ml_collections.ConfigDict()
	data.p , data.q = 32, 32
	data.channels = 3
	data.dataset = "SVHN"

	# distributions 
	config.dist = dist = ml_collections.ConfigDict()
	dist.latent = "Gaussian"
	dist.latent_dim = 50
	dist.min_std = 1e-2
	dist.data = "DiscNormal"

	#base VAE optimization
	config.optim = optim = ml_collections.ConfigDict()
	optim.lr = 1e-4
	optim.optimizer = "Adam"
	optim.epochs = 100

	#QAVI hyperparameters
	config.qavi = qavi = ml_collections.ConfigDict()
	qavi.beta = 20
	qavi.batches = 1000
	qavi.K = 100
	qavi.iterations = 300
    
	return config

