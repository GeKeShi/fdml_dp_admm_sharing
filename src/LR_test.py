## TO DO
# data loading and initialization
# training
# recording and saving
# noise adding and handling

from config import config
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy import sparse
import time
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, plot_roc_curve
from matplotlib import pyplot as plt
if config["is_parallel"] == True:
	from multiprocessing import current_process, Pool 
	pool = Pool()

import util
import itertools
# reload(util)
np.random.seed(0)

def fast_exp(x):
	pass

class YYB_Dataset(Dataset):
	"""yyb dataset."""

	def __init__(self, train_path, test_path, transform=None):
		"""
		Args:
			data_path: str, path to data file
			transform: None
		"""
		train_feature, train_label, test_feature, test_label= util.load_svmlightfile_data(train_path, test_path)
		self.train_feature = train_feature
		# print(train_feature[0:2,:])
		self.train_label = train_label
		# print(train_label[0:2])
		self.test_feature = test_feature
		self.test_label = test_label
		self.size_feature = train_feature.shape[1]
		# print("{} size feature {}".format(self.train_feature.shape[0], self.size_feature))
		self.train_size = self.train_feature.shape[0]
		self.test_size = self.test_feature.shape[0]
		self.transform = transform

	def __len__(self):
		return self.train_size

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample = {'feature': self.train_feature[idx], 'lable': self.train_label[idx]}

		if self.transform:
			sample = self.transform(sample)

		return sample




class server():
	def __init__(self, config):
		self.config = config
		# loading data
		train_path = os.path.join(self.config["input_dir_path"], "raw_train")
		test_path = os.path.join(self.config["input_dir_path"], "raw_test")
		self.yyb_data = YYB_Dataset(train_path,test_path)
		
		
	def getminibatch(self, feature_iter, label_iter, batch_size):
		batched_feature = itertools.islice(feature_iter, batch_size)
		dense_data = np.array([item.toarray().flatten() for item in batched_feature])
		# print("data shape is {}".format(dense_data.shape))
		batched_label = itertools.islice(label_iter, batch_size)
		dense_label = np.array(list(batched_label))
		# print("label shape is {}".format(dense_label.shape))
		return dense_data, dense_label


	def iterminibatch(self, feature_iter, label_iter, batch_size):
		batched_feature, batched_label = self.getminibatch(feature_iter, label_iter, batch_size)
		# change iterable to list to use len(), final batch is []
		while len(list(batched_feature)):
			yield batched_feature, batched_label
			batched_feature, batched_label = self.getminibatch(feature_iter, label_iter, batch_size)	

	def train(self):
		# LRmodel = LogisticRegression(random_state=0).fit(self.train_feature, self.train_label)
		LRmodel = SGDClassifier(loss='log', random_state=0)
		for iteration in range(self.config['max_iter']):
			# iter() the training set to make it iterable so as to get different batched data 
			self.train_data_iter = self.iterminibatch(iter(self.yyb_data.train_feature), iter(self.yyb_data.train_label), self.config["batch_size"])
			for i_batch, (batched_feature, batched_label) in enumerate(self.train_data_iter):
				LRmodel.partial_fit(batched_feature, batched_label, classes=np.array([0.0, 1.0]))
				batch_result = LRmodel.predict(batched_feature)
				if i_batch%10 == 0:
					print('iteration {} batch {} training_loss {}'.format(iteration, i_batch, util.logloss_logit_form(batched_label, batch_result)))
			
		
		# train_score = LRmodel.predict_proba(self.train_feature)
		# test_score = LRmodel.predict_proba(self.test_feature)
		test_score = LRmodel.predict(self.yyb_data.test_feature)
		# print(test_score)
		# print("test score {}".format(roc_auc_score(self.test_label, test_score[:,0])))
		# print("test score {}".format(roc_auc_score(self.test_label, test_score[:,1])))
		print('testing loss{}'.format(util.logloss_logit_form(self.yyb_data.test_label, test_score)))

		# print("train score {}".format(roc_auc_score(self.train_label, train_score)))
		# LRmodel_plot = plot_roc_curve(LRmodel, self.test_feature, self.test_label)
		# plt.show()




if __name__ == "__main__":
	# input_dir_path noise_scale num_workers
	if len(sys.argv) > 1:
		if sys.argv[1] == "h":
			print("input_dir_path noise_scale num_workers maxiteration [epsilon delta]")
			exit(1)
		print("Using command line parameters")
		try:
			config["input_dir_path"] = sys.argv[1]
			config["noise_scale"] = float(sys.argv[2])
			config["num_workers"] = int(sys.argv[3])
			config["max_iter"] = int(sys.argv[4])
		except:
			print("Wrong input parameters. Use option h for help.")
			exit(-1)
		if len(sys.argv) > 5:
			try:
				config["epsilon"] = float(sys.argv[5])
				config["delta"] = float(sys.argv[6])
				config["noise_eval_method"] = "computed"
			except:
				print("Wrong input parameters. Use option h for help.")
				exit(-1)
	else:
		print("Using default parameters in config")
	test = server(config)
	test.train()
