import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import time

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


from data import *
from model import HGCN



def train(epoch):
	model.train()
	optimizer.zero_grad()
	logits, _ = model(ft_dict, adj_dict)

	logits = F.log_softmax(logits[target_type], dim=1)
	idx_train = label[target_type][1]
	x_train = logits[idx_train]
	y_train = label[target_type][0][idx_train]
	loss_train = F.nll_loss(x_train, y_train)
	f1_micro_train = f1_score(y_train.data.cpu(), x_train.data.cpu().argmax(1), average='micro')
	f1_macro_train = f1_score(y_train.data.cpu(), x_train.data.cpu().argmax(1), average='macro')

	loss_train.backward()
	optimizer.step()



	'''///////////////// Validatin ///////////////////'''
	if epoch % 10 == 0:
			
		model.eval()
		logits, _ = model(ft_dict, adj_dict)

		logits = F.log_softmax(logits[target_type], dim=1)
		idx_val = label[target_type][2]
		x_val = logits[idx_val]
		y_val = label[target_type][0][idx_val]
		f1_micro_val = f1_score(y_val.data.cpu(), x_val.data.cpu().argmax(1), average='micro')
		f1_macro_val = f1_score(y_val.data.cpu(), x_val.data.cpu().argmax(1), average='macro')
		
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1: {:.4f}'.format(f1_micro_train.item()),
			  'train macro f1: {:.4f}'.format(f1_macro_train.item()),
			  'val micro f1: {:.4f}'.format(f1_micro_val.item()),
			  'val macro f1: {:.4f}'.format(f1_macro_val.item()),
			 )


def test():
	model.eval()
	logits, embd = model(ft_dict, adj_dict)

	logits = F.log_softmax(logits[target_type], dim=1)
	idx_test = label[target_type][3]
	x_test = logits[idx_test]
	y_test = label[target_type][0][idx_test]
	f1_micro_test = f1_score(y_test.data.cpu(), x_test.data.cpu().argmax(1), average='micro')
	f1_macro_test = f1_score(y_test.data.cpu(), x_test.data.cpu().argmax(1), average='macro')
	
	print(
  		  'test micro f1: {:.4f}'.format(f1_micro_test.item()),
		  'test macro f1: {:.4f}'.format(f1_macro_test.item()),
		 )


	return (f1_micro_test, f1_macro_test)



if __name__ == '__main__':

	cuda = True # Enables CUDA training.
	lr = 0.01 # Initial learning rate.c
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean
	train_percent = 0.2
	

	run_num = 2
	for run in range(run_num):
		t_start = time.time()

		seed = run
		np.random.seed(seed)
		torch.manual_seed(seed)
		if cuda and torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)

		print('\nHGCN run: ', run)
		print('train percent: ', train_percent)
		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)



		hid_layer_dim = [64,32,16,8]
		epochs = 250    # classfication best epoch
		label, ft_dict, adj_dict = load_imdb_3228(train_percent)

		# hid_layer_dim = [64,32,16,8]
		# epochs = 130    # classfication best epoch
		# label, ft_dict, adj_dict = load_acm_4025(train_percent)

		# hid_layer_dim = [64,32,16,8]
		# epochs = 200      # classfication best epoch
		# label, ft_dict, adj_dict = load_dblp_4057(train_percent)

		# hid_layer_dim = [64,32,16,8]
		# epochs = 100    # classfication best epoch
		# label, ft_dict, adj_dict = load_odbmag_4017(train_percent)


		print('all object types: ', list(ft_dict.keys()))
		target_type = list(label.keys())[0]
		print('target object type: ', target_type)
		class_num = np.unique(label[target_type][0]).shape[0]
		print('number of classes: ', class_num, '\n')


		layer_shape = []
		
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		output_layer_shape = dict.fromkeys(ft_dict.keys(), class_num)
		
		layer_shape.append(input_layer_shape)
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)


		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
		model = HGCN(
					net_schema=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()),
					type_fusion=type_fusion,
					type_att_size=type_att_size,
					)

		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


		if cuda and torch.cuda.is_available():
			model.cuda()

			for k in ft_dict:
				ft_dict[k] = ft_dict[k].cuda()
			for k in adj_dict:
				for kk in adj_dict[k]:
					adj_dict[k][kk] = adj_dict[k][kk].cuda()
			for k in label:
				for i in range(len(label[k])):
					label[k][i] = label[k][i].cuda()

		for epoch in range(epochs):
			train(epoch)
		print('\nfinish training, start test!\n')
		(micro_f1, macro_f1) = test()

		t_end = time.time()
		print('Total time: ', t_end - t_start)