import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from mask_generator.mask_generator import RandomMask, BatchRandomMask
from torch.utils.data import Dataset
import pickle
from sklearn.datasets import load_iris, load_breast_cancer, load_boston 
from PIL import Image
from utils import *

def load_file(filename):
	with open(filename + ".pkl", 'rb') as file:
		images = pickle.load(file)
	with open(filename + "_labels.pkl", 'rb') as file:
		labels = pickle.load(file)

	return images, labels

def save_file(filename, images, labels): 
	s_file(filename + ".pkl", images)
	s_file(filename + "_labels.pkl", labels)

def normalize(images):
	images = (images - 0.5)/0.5
	return images

def un_normalize(images):
	images = images*0.5 + 0.5
	return images

def one_hot_labels(x):
	b = np.zeros((x.shape[0], x.max() + 1))
	b[np.arange(x.shape[0]), x] = 1
	return b

def generate_patches(xmiss, shape, patch_size=10):
	[n, channels, p ,q] = shape
	num_patches = 2 
	patch_size_p = patch_size 
	patch_size_q = patch_size 

	miss_pattern_x = [np.random.choice((p - patch_size_p), num_patches, replace=False) for i in range(n)]
	miss_pattern_y = [np.random.choice((q - patch_size_q), num_patches, replace=False) for i in range(n)]

	for i in range(n):
		for a,b in zip(miss_pattern_y[i], miss_pattern_x[i]):
			start_x = int(a)
			end_x = int(a + patch_size_q)
			start_y = int(b)
			end_y = int(b + patch_size_p)
			xmiss[i,:,start_x: end_x, start_y:end_y] = np.nan

	mask = np.isfinite(xmiss)  # False indicates missing, True indicates observed
	return mask

def generate_TH(shape):
	(n , channels, p, q) = shape
	mask = np.zeros((n,p,q))

	for i in range(n):
		angle = np.random.choice(360, 1)
		mask_one = np.ones((p+30,q+30))
		mask_one[int((p+30)/2):,:] = 0
		
		im = Image.fromarray(mask_one)
		im = im.rotate(angle)

		left = (p+30 - p)/2
		top = (q+30 - q)/2
		right = (p+30 + p)/2
		bottom = (q+30 + q)/2

		# Crop the center of the image
		im = im.crop((left, top, right, bottom))

		mask[i] = np.array(im)
	
	mask = np.repeat(mask.reshape([n,1,p,q]), channels, axis=1) 
	return mask


def random_miss(shape ,n_miss=0):
	[n, channels, p, q] = shape
	mask = np.ones((n,p,q))
	mask_flat = mask.flatten()

	miss_pattern = [(i)*p*q + np.random.choice(p*q, n_miss, replace=False) for i in range(n)]
	miss_pattern = np.asarray(miss_pattern).astype(np.int)
	mask_flat[miss_pattern] = 0 # np.nan
	mask = mask_flat.reshape([n,1, p,q])

	mask = np.repeat(mask, channels, axis=1)
	return mask


def random_miss_uci(testset, n_miss):
	xhat_0 = np.copy(testset)
	n_test, p = np.shape(testset)
	mask = np.ones((n_test, p))

	for i in range(n_test):
		miss_pattern = np.random.choice(p, np.floor(n_miss).astype(np.int), replace=False)
		for j in range(len(miss_pattern)):
			xhat_0[i,miss_pattern[j]] = 0
			mask[i,miss_pattern[j]] = 0  

	return xhat_0, mask


class load_dataset(Dataset):
	def __init__(self, dataset, data_type="mnist", miss_string=None, train=False):
		(self.images,self.labels) = dataset
		if train: mode = "train"
		else: mode = "test"

		if len(self.images.shape) == 3: 
			[n, p, q] = self.images.shape
			self.images = self.images.reshape([n, 1 , p, q])

		self.labels = one_hot_labels(self.labels)
		[n, channels, p, q] = self.images.shape
		self.masks = np.ones([n,channels,p,q]).astype(np.bool)

		xmiss = np.copy(self.images).astype(np.float)
		#xmiss_flat = xmiss.flatten()

		if not train: 
			if miss_string=='patches': np.random.seed(1234)
			else: np.random.seed(5678)
		else: np.random.seed(3367)

		if miss_string == 'patches': 
			if data_type == "mnist" : patch_size = 10
			else: patch_size = 15
			self.masks = generate_patches(xmiss, xmiss.shape, patch_size)
		elif miss_string == 'rotating_half':
			self.masks = generate_TH(xmiss.shape)
		elif miss_string == '50random':
			self.masks = random_miss(n,p,q,int(p*q/2))
		elif miss_string == '10random':
			self.masks = random_miss(n,p,q,10*10*2)
		elif miss_string == 'random':       
			file_mask = data_type+"/data/randommask_" + data_type + "_" + mode + ".pkl" 
			if os.path.exists(file_mask):
				with open(file_mask, 'rb') as file:
					self.masks = pickle.load(file)
			else:
				self.masks = BatchRandomMask(n, p, (0., 1.)) 
				with open(file_mask, 'wb') as file:
					pickle.dump(self.masks, file)
			self.masks = np.repeat(self.masks, channels, axis=1)

		self.masks = self.masks.reshape(n,channels,p,q).astype(np.bool)
		self.xhats_0 = np.copy(self.images).reshape(n,channels,p,q).astype(np.float)
		self.xhats_0[~self.masks] = 0 

	def __getitem__(self, idx):
		return self.xhats_0[idx], self.masks[idx], self.images[idx] ,  self.labels[idx]

	def __len__(self):	
		return len(self.images)

def dataset_loader(data_dir = "data", batch_size=64, miss_string=None, mode="train", data_type ='mnist'):
	data_dir = data_type + "/" + data_dir
	if mode == "train": istraining = True
	else: istraining = False

	file_name = data_dir + "/" + mode + "_permuted"

	if os.path.exists(file_name + ".pkl"):
		print("Loaded from permuted data")
		images, labels = load_file(file_name)
	else:
		if data_type =='mnist' : dataset = datasets.MNIST(root=data_dir, train=istraining, download=True, transform=transforms.ToTensor())
		else : dataset = datasets.SVHN(root=data_dir, split = mode, download=True, transform=transforms.ToTensor())

		seq = torch.randperm(dataset.data.shape[0])
		images = dataset.data[seq]
		if data_type =='mnist' : labels = dataset.targets[seq]
		else : labels = dataset.labels[seq]
		save_file(file_name, images, labels)

	if mode == "test":
		images = images[:1000]
		labels = labels[:1000]

	images = images/255
	if data_type == 'svhn' : images = normalize(images)

	data_loader = torch.utils.data.DataLoader(dataset=load_dataset((images,labels), data_type=data_type, miss_string=miss_string, train=istraining),batch_size=batch_size)
	return data_loader

def train_valid_split(xfull, s=0.65, t=0.8):
	np.random.shuffle(xfull)
	n = np.shape(xfull)[0] # number of observations
	p = np.shape(xfull)[1] # number of features

	split1 = int(s*n)
	split2 = int(t*n)
	trainset = xfull[:split1]
	validset = xfull[split1:split2]
	testset = xfull[split2:]

	return trainset, validset, testset

def save_uci_files(dataset):
	if dataset=='breast_cancer':
		data = load_breast_cancer()['data']
	elif dataset == "red_wine":
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
		data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
	elif dataset == "white_wine":
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
		data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
	elif dataset =='banknote':
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
		data = np.array(pd.read_csv(url, low_memory=False, sep=','))[:,0:4]
	elif dataset =='yeast':
		data = np.loadtxt(open("data/yeast.csv", "rb"), delimiter=",", skiprows=0)
	elif dataset =='concrete':
		data = np.loadtxt(open("data/concrete.csv", "rb"), delimiter=",", skiprows=0)

	#Normalize the data
	xfull = (data - np.mean(data,0))/np.std(data,0)
	
	trainset, validset, testset = train_valid_split(xfull)

	s_file(os.getcwd() + "/data/new_uci/" + dataset + "-train.pkl", trainset)
	s_file(os.getcwd() + "/data/new_uci/" + dataset + "-valid.pkl", validset)
	s_file(os.getcwd() + "/data/new_uci" + dataset + "-testset.pkl", testset)

	return trainset, validset, testset

def load_uci_datasets(dataset='breast_cancer', missing_for_train = False, perc_miss=0.5):

	if os.path.exists(os.getcwd() + "/data/new_uci/" + dataset + "-train.pkl"): 
		trainset = r_file(os.getcwd() + "/data/new_uci/" + dataset + "-train.pkl")
		validset = r_file(os.getcwd() + "/data/new_uci/" + dataset + "-valid.pkl")
		testset = r_file(os.getcwd() + "/data/new_uci/" + dataset + "-testset.pkl")
	else: 
		trainset, validset, testset = save_uci_files(dataset)

	np.random.seed(1234)
	n_test, p = np.shape(testset)

	##For missing data in testset
	if perc_miss==2: n_miss = 2
	else: n_miss = int(perc_miss*p)
	xhat_0, mask = random_miss_uci(testset, n_miss)

	if missing_for_train:
		train_miss, mask_train = random_miss_uci(trainset, n_miss)
		valid_miss, mask_valid = random_miss_uci(validset, n_miss)
		return train_miss, trainset, mask_train, valid_miss, mask_valid, xhat_0, mask

	return trainset, validset, xhat_0, mask, testset

