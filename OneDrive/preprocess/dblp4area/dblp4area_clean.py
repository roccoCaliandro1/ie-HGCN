import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer



# paper feature
p_raw_id_feature = np.loadtxt('./cleaned/paper.txt', delimiter='\t', dtype=np.str)

corpus = p_raw_id_feature[:,1]
vectorizer = HashingVectorizer(n_features=2**7)
p_text_ft = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
p_text_ft = transformer.fit_transform(p_text_ft)

p_raw_ids = p_raw_id_feature[:,0].astype(np.int32)
p_num = len(np.unique(p_raw_ids))
if len(p_raw_ids) == p_num:
	print('number of unique paper: ', len(p_raw_ids))
	p_id_raw2int = dict(zip(p_raw_ids, list(range(p_num))))
	p_ft = p_text_ft.todense()
	print('paper feature dimension: ', p_ft.shape)
else:
	print('duplicate paper id feature!')
	exit()

# paper label
p_raw_label = np.loadtxt('./cleaned/paper_label.txt', delimiter='\t', dtype=np.str)
p_raw_id_label = p_raw_label[:,:2].astype(np.int32)
p_label = np.full((len(p_id_raw2int), ), -1, dtype=np.int32)
for item in p_raw_id_label:
	p_label[p_id_raw2int[item[0]]] = item[1]
print('paper label num: ', np.where(p_label != -1)[0].shape[0], '\n')


# author id name
a_raw_id_name = np.loadtxt('./cleaned/author.txt', delimiter='\t', dtype=np.str)
a_raw_ids = a_raw_id_name[:,0].astype(np.int32)
a_num = len(np.unique(a_raw_ids))
if len(a_raw_ids) == a_num:
	print('number of unique author: ', len(a_raw_ids))
	a_id_raw2int = dict(zip(a_raw_ids, list(range(a_num))))
else:
	print('duplicate author id!')
	exit()

# author label
a_raw_label = np.loadtxt('./cleaned/author_label.txt', delimiter='\t', dtype=np.str)
a_raw_id_label = a_raw_label[:,:2].astype(np.int32)
a_label = np.full((len(a_id_raw2int), ), -1, dtype=np.int32)
for item in a_raw_id_label:
	a_label[a_id_raw2int[item[0]]] = item[1]
print('author label num: ', np.where(a_label != -1)[0].shape[0], '\n')



# conf id name
c_raw_id_name = np.loadtxt('./cleaned/conf.txt', delimiter='\t', dtype=np.str)
c_raw_ids = c_raw_id_name[:,0].astype(np.int32)
c_num = len(np.unique(c_raw_ids))
if len(c_raw_ids) == c_num:
	print('number of unique conf: ', len(c_raw_ids))
	c_id_raw2int = dict(zip(c_raw_ids, list(range(c_num))))
else:
	print('duplicate conf id!')
	exit()

# conf label
c_raw_label = np.loadtxt('./cleaned/conf_label.txt', delimiter='\t', dtype=np.str)
c_raw_id_label = c_raw_label[:,:2].astype(np.int32)
c_label = np.full((len(c_id_raw2int), ), -1, dtype=np.int32)
for item in c_raw_id_label:
	c_label[c_id_raw2int[item[0]]] = item[1]
print('conf label num: ', np.where(c_label != -1)[0].shape[0], '\n')



# term id name
t_raw_id_name = np.loadtxt('./cleaned/term.txt', delimiter='\t', dtype=np.str)
t_raw_ids = t_raw_id_name[:,0].astype(np.int32)
t_num = len(np.unique(t_raw_ids))
if len(t_raw_ids) == t_num:
	print('number of unique term: ', len(t_raw_ids), '\n')
	t_id_raw2int = dict(zip(t_raw_ids, list(range(t_num))))
else:
	print('duplicate term id!')
	exit()



# paper - author
p_a = np.loadtxt('./cleaned/paper_author.txt', delimiter='\t', dtype=np.int32)
row = [p_id_raw2int[p_id] for p_id in p_a[:,0]]
col = [a_id_raw2int[a_id] for a_id in p_a[:,1]]
data = np.ones(p_a.shape[0], dtype=np.float32)
sp_A_p_a = sp.coo_matrix((data, (row, col)), shape=(p_num, a_num))
print('sparse mat paper author shape: ', sp_A_p_a.shape, '\n')



# paper - conf
p_c = np.loadtxt('./cleaned/paper_conf.txt', delimiter='\t', dtype=np.int32)
row = [p_id_raw2int[p_id] for p_id in p_c[:,0]]
col = [c_id_raw2int[c_id] for c_id in p_c[:,1]]
data = np.ones(p_c.shape[0], dtype=np.float32)
sp_A_p_c = sp.coo_matrix((data, (row, col)), shape=(p_num, c_num))
print('sparse mat paper conf shape: ', sp_A_p_c.shape, '\n')



# paper - term
p_t = np.loadtxt('./cleaned/paper_term.txt', delimiter='\t', dtype=np.int32)
row = [p_id_raw2int[p_id] for p_id in p_t[:,0]]
col = [t_id_raw2int[t_id] for t_id in p_t[:,1]]
data = np.ones(p_t.shape[0], dtype=np.float32)
sp_A_p_t = sp.coo_matrix((data, (row, col)), shape=(p_num, t_num))
print('sparse mat paper term shape: ', sp_A_p_t.shape, '\n')



# with open('./dump/dblp4area_ids_map_dict.pkl', 'wb') as out_file:
# 	pickle.dump((p_id_raw2int, a_id_raw2int, c_id_raw2int, t_id_raw2int), out_file)

# with open('./dump/dblp4area_paper_feature.pkl', 'wb') as out_file:
# 	pickle.dump(p_ft, out_file)

# with open('./dump/dblp4area_label.pkl', 'wb') as out_file:
# 	pickle.dump((p_label, a_label, c_label), out_file)

# with open('./dump/dblp4area_sp_adj_mats.pkl', 'wb') as out_file:
# 	pickle.dump((sp_A_p_a, sp_A_p_c, sp_A_p_t), out_file)



# with open('./dump/dblp4area_ids_map_dict.pkl', 'rb') as in_file:
# 	(p_id_raw2int, a_id_raw2int, c_id_raw2int, t_id_raw2int) = pickle.load(in_file)

# with open('./dump/dblp4area_paper_feature.pkl', 'rb') as in_file:
# 	p_ft = pickle.load(in_file)

# with open('./dump/dblp4area_label.pkl', 'rb') as in_file:
# 	(p_label, a_label, c_label) = pickle.load(in_file)

# with open('./dump/dblp4area_sp_adj_mats.pkl', 'rb') as in_file:
# 	(sp_A_p_a, sp_A_p_c, sp_A_p_t) = pickle.load(in_file)
