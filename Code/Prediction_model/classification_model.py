import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim

class SimpleMLP(nn.Module):

	def __init__(self,
				in_size,
				hidden_size,
				out_size = 14, 
				dropout = 0.1,num_hid_layers=5):
		super(SimpleMLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(in_size, hidden_size)
		self.relu = nn.ReLU()
		self.inter_layer1 = nn.Linear(hidden_size, hidden_size)
		self.hid = [nn.Linear(hidden_size, hidden_size) for i in range(num_hid_layers)]
		self.w2 = nn.Linear(hidden_size, out_size)
		self.dropout = nn.Dropout(p = dropout)
		self.num_hid_layers = num_hid_layers
		self.initialize()

	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.inter_layer1.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		self.inter_layer1.bias.data.zero_()
		# print(self.w1, self.w2)

	def forward(self, x):

		h1 = self.w1(x)
		a = self.relu(h1)
		# for layer in range(self.num_hid_layers):
		# 	a=self.hid[layer](a)
		# 	a= self.relu(a)
		# 	a = self.dropout(a)
		a = self.dropout(a)
		a = self.inter_layer1(a)
		a = self.relu(a)
		a = self.dropout(a)
		h2 = self.w2(a)
		#print(h2.shape)
		return h2








class JotilMLP(nn.Module):

	def __init__(self,
				in_size,
				hidden_size,
				out_size = 14, 
				dropout = 0.1,num_hid_layers=5):
		super(SimpleMLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(in_size, hidden_size)
		self.relu = nn.ReLU()
		self.inter_layer1 = nn.Linear(hidden_size, hidden_size)
		self.hid = [nn.Linear(hidden_size, hidden_size) for i in range(num_hid_layers)]
		self.w2 = nn.Linear(hidden_size, out_size)
		self.dropout = nn.Dropout(p = dropout)
		self.num_hid_layers = num_hid_layers
		self.initialize()

	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.inter_layer1.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		self.inter_layer1.bias.data.zero_()
		# print(self.w1, self.w2)

	def forward(self, x):

		h1 = self.w1(x)
		a = self.relu(h1)
		# for layer in range(self.num_hid_layers):
		# 	a=self.hid[layer](a)
		# 	a= self.relu(a)
		# 	a = self.dropout(a)
		a = self.dropout(a)
		a = self.inter_layer1(a)
		a = self.relu(a)
		a = self.dropout(a)
		h2 = self.w2(a)
		#print(h2.shape)
		return h2