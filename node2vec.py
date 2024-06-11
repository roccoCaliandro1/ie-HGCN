#from gensim.models import Word2Vec

import scipy
import gensim

print(scipy.__version__)

size = 128

seed=123
window=10
min_count=0
sg=1
workers=5 # see if we can increase it
epochs = 10


w2v_model = Word2Vec(vector_size=size, seed=seed, window=window, 
                     min_count=min_count, sg=sg, workers=workers)

# vocabulary creation
# token_word = lista di token (lista di liste)
token_word = [['ciao'], ['poi']]
w2v_model.build_vocab(token_word, min_count=1)
total_examples = w2v_model.corpus_count


w2v_model.train(token_word, total_examples=total_examples, epochs=epochs)

# w2v_model.wv.key_to_index["ciao"] # torna il dizionario ad ogni parola (k) associa un indice 
# per prendere l'embedding di ciao, m.model.wv['ciao']