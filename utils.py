import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def s_file(filename, x, mode="wb"):
	with open(filename, mode) as file:
		pickle.dump(x, file)

def r_file(filename, mode="rb"): 
	with open(filename, mode) as file:
		x = pickle.load(file)   
	return x 
           
def read_file(filename, num_rows):
	x = []
	with open(filename, 'rb') as file:
		for j in range(num_rows) : x.append(pickle.load(file))  
	return x           

def check_file(filename, num_lines):
	i=0
	with open(filename, 'rb') as f:
		while True:
			try:
				data = pickle.load(f)
				i+=1
			except EOFError as exc:
				return i == num_lines
            
def del_extra(filename, num_lines):
	i=0
	total_data = []
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		total_data.append(data)
        
	print(len(total_data), total_data)
	os.remove(filename)
	with open(filename, 'ab+') as f:
		for j in range(num_lines): pickle.dump(total_data[j], f)                    
	return 

def approx_latent_loss(p_z, q_z, z, params):
	##Calculate KL using samples
	batch_size = z.shape[1]
	K = z.shape[0]
	logqz = q_z.log_prob(z).reshape(K,batch_size)
	logpz = p_z.log_prob(z).reshape(K,batch_size)    
	k_l = torch.mean(logqz - logpz, 0)
	return k_l           
            
def data_replicate(data, K):
	if len(data.shape)==2: data_batch = torch.Tensor.repeat(data,[K,1]) 
	else: data_batch = torch.Tensor.repeat(data,[K,1,1,1]) 
	return data_batch

def jth_batch( data, j, batch_size):
	return  data[int(j*batch_size) : int((j+1)*batch_size) ]


def show_images(b_data, b_full, imputations, data='mnist'):
	[K, batch_size, channels,p,q] = imputations.shape
	fig = plt.figure(figsize=(6, 6))
	# setting values to rows and column variables
	rows = 10 #n_images
	columns = 10 #K imputations

	for i in range(rows):
		for j in range(columns):
			fig.add_subplot(rows, columns, i*columns + j +1)
			# showing image
			if data=='mnist': 
				if j == 0: a = np.squeeze(b_full[i].cpu().data.numpy().reshape(1,1,28,28))
				elif j == 1: a = np.squeeze(b_data[i].cpu().data.numpy().reshape(1,1,28,28))
				else : a = np.squeeze(imputations[j,i].cpu().data.numpy().reshape(1,1,28,28))
				plt.imshow(a, cmap='gray', vmin=0, vmax=1)
			else: 
				#print(i,j)
				#print(imputation_miss[i,j].shape)
				a = np.transpose(expit(imputation_miss[j,i].cpu().data.numpy()), (1,2,0))
				plt.imshow(a)
			plt.axis('off')
	plt.show()
	plt.close()

def mse_(xhat, xtrue, data='mnist'): # MSE function for imputations
	if data != 'mnist':
		xhat = np.array(xhat*0.5 + 0.5)
		xtrue = np.array(xtrue*0.5 + 0.5)
	num_missing = xhat.shape[1]
	return np.sum(np.power(xhat-xtrue,2),1) #/num_missing

