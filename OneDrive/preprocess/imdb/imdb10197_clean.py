import numpy as np
import pickle
import scipy.sparse as sp
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
movie_file_path = os.path.join(current_dir, 'cleaned', 'movie.txt')


m_raw_id_feature = np.genfromtxt(movie_file_path, delimiter='\t', missing_values='\\N', filling_values=0.0, dtype=np.float32)
print('movie id feature shape: ', m_raw_id_feature.shape)
m_raw_ids = m_raw_id_feature[:,0].astype(np.int32)
m_num = len(np.unique(m_raw_ids))
if m_num == len(m_raw_ids):
	print('number of unique movie: ', m_num, '\n')
	m_id_raw2int = dict(zip(m_raw_ids, list(range(m_num))))
	m_ft = m_raw_id_feature[:,1:]
else:
	print('duplicate paper id feature!\n')
	exit()


movie_actor_path = os.path.join(current_dir, 'cleaned', 'movie_actor.txt')
m_a = np.genfromtxt(movie_actor_path, delimiter='\t', dtype=str)
print('movie_actor shape: ', m_a.shape)
a_raw_ids = m_a[:,1]
a_unique_raw_ids = np.unique(a_raw_ids)
a_id_raw2int = dict(zip(a_unique_raw_ids, list(range(len(a_unique_raw_ids)))))
a_num = len(a_unique_raw_ids)
print('number of unique actor: ', a_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_a[:,0].astype(np.int32)]
col = [a_id_raw2int[a_id] for a_id in m_a[:,1]]
data = m_a[:,2].astype(np.float32)
sp_A_m_a = sp.coo_matrix((data, (row, col)), shape=(m_num, a_num))


movie_country_path = os.path.join(current_dir, 'cleaned', 'movie_country.txt')
m_c = np.genfromtxt(movie_country_path, delimiter='\t', dtype=str)
print('movie_country shape: ', m_c.shape)
c_raw_ids = m_c[:,1]
c_unique_raw_ids = np.unique(c_raw_ids)
c_id_raw2int = dict(zip(c_unique_raw_ids, list(range(len(c_unique_raw_ids)))))
c_num = len(c_unique_raw_ids)
print('number of unique country: ', c_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_c[:,0].astype(np.int32)]
col = [c_id_raw2int[c_id] for c_id in m_c[:,1]]
data = np.ones(m_c.shape[0], dtype=np.float32)
sp_A_m_c = sp.coo_matrix((data, (row, col)), shape=(m_num, c_num))


movie_director_path = os.path.join(current_dir, 'cleaned', 'movie_director.txt')
m_d = np.genfromtxt(movie_director_path, delimiter='\t', dtype=str)
print('movie_director shape: ', m_d.shape)
d_raw_ids = m_d[:,1]
d_unique_raw_ids = np.unique(d_raw_ids)
d_id_raw2int = dict(zip(d_unique_raw_ids, list(range(len(d_unique_raw_ids)))))
d_num = len(d_unique_raw_ids)
print('number of unique director: ', d_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_d[:,0].astype(np.int32)]
col = [d_id_raw2int[d_id] for d_id in m_d[:,1]]
data = np.ones(m_d.shape[0], dtype=np.float32)
sp_A_m_d = sp.coo_matrix((data, (row, col)), shape=(m_num, d_num))

movie_tag_path = os.path.join(current_dir, 'cleaned', 'movie_tag.txt')
m_t = np.genfromtxt(movie_tag_path, delimiter='\t', dtype=np.int32)
print('movie_tag shape: ', m_t.shape)
t_raw_ids = m_t[:,1]
t_unique_raw_ids = np.unique(t_raw_ids)
t_id_raw2int = dict(zip(t_unique_raw_ids, list(range(len(t_unique_raw_ids)))))
t_num = len(t_unique_raw_ids)
print('number of unique tag: ', t_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_t[:,0]]
col = [t_id_raw2int[t_id] for t_id in m_t[:,1]]
data = m_t[:,2].astype(np.float32)
sp_A_m_t = sp.coo_matrix((data, (row, col)), shape=(m_num, t_num))


movie_user_path = os.path.join(current_dir, 'cleaned', 'movie_user.txt')
m_u = np.genfromtxt(movie_user_path, delimiter='\t', dtype=np.float32)
print('movie_user shape: ', m_u.shape)
u_raw_ids = m_u[:,1].astype(np.int32)
u_unique_raw_ids= np.unique(u_raw_ids)
u_id_raw2int = dict(zip(u_unique_raw_ids, list(range(len(u_unique_raw_ids)))))
u_num = len(u_unique_raw_ids)
print('number of unique user: ', u_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_u[:,0].astype(np.int32)]
col = [u_id_raw2int[u_id] for u_id in m_u[:,1].astype(np.int32)]
data = m_u[:,2]
sp_A_m_u = sp.coo_matrix((data, (row, col)), shape=(m_num, u_num))


movie_genre_path = os.path.join(current_dir, 'cleaned', 'movie_genre.txt')
m_g = np.genfromtxt(movie_genre_path, delimiter='\t', dtype=str)
print('movie_genre shape: ', m_g.shape)
g_raw_ids = m_g[:,1]
g_unique_raw_ids = np.unique(g_raw_ids)
g_id_raw2int = dict(zip(g_unique_raw_ids, list(range(len(g_unique_raw_ids)))))
g_num = len(g_unique_raw_ids)
print('number of unique genre: ', g_num, '\n')
row = [m_id_raw2int[m_id] for m_id in m_g[:,0].astype(np.int32)]
col = [g_id_raw2int[g_id] for g_id in m_g[:,1]]
data = np.ones(m_g.shape[0], dtype=np.float32)
sp_A_m_g = sp.coo_matrix((data, (row, col)), shape=(m_num, g_num))


with open('./dump/imdb10197_ids_map_dict.pkl', 'wb') as out_file:
	pickle.dump((m_id_raw2int, a_id_raw2int, c_id_raw2int, d_id_raw2int, t_id_raw2int, u_id_raw2int, g_id_raw2int), out_file)

with open('./dump/imdb10197_movie_feature.pkl', 'wb') as out_file:
	pickle.dump(m_ft, out_file)

with open('./dump/imdb10197_sp_adj_mats.pkl', 'wb') as out_file:
	pickle.dump((sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g), out_file)



with open('./dump/imdb10197_ids_map_dict.pkl', 'rb') as in_file:
	(m_id_raw2int, a_id_raw2int, c_id_raw2int, d_id_raw2int, t_id_raw2int, u_id_raw2int, g_id_raw2int) = pickle.load(in_file)
print('number of movie: ', len(m_id_raw2int))
print('number of actor: ', len(a_id_raw2int))
print('number of country: ', len(c_id_raw2int))
print('number of director: ', len(d_id_raw2int))
print('number of tag: ', len(t_id_raw2int))
print('number of user: ', len(u_id_raw2int))
print('number of genre: ', len(g_id_raw2int))

with open('./dump/imdb10197_movie_feature.pkl', 'rb') as in_file:
	m_ft = pickle.load(in_file)
print('movie feature shape: ', m_ft.shape, type(m_ft), m_ft.dtype)

with open('./dump/imdb10197_sp_adj_mats.pkl', 'rb') as in_file:
	(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)

print('m_a: ', sp_A_m_a.max(), sp_A_m_a.shape, type(sp_A_m_a))
print('m_u: ', sp_A_m_u.max(), sp_A_m_u.shape, type(sp_A_m_u))
print('m_t: ', sp_A_m_t.max(), sp_A_m_t.shape, type(sp_A_m_t))
print('m_d: ', sp_A_m_d.max(), sp_A_m_d.shape, type(sp_A_m_d))
print('m_c: ', sp_A_m_c.max(), sp_A_m_c.shape, type(sp_A_m_c))
print('m_g: ', sp_A_m_g.max(), sp_A_m_g.shape, type(sp_A_m_g))
