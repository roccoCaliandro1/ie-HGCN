import os
from gensim.models import Word2Vec
import gensim
import pandas as pd
import pickle
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def trainWord2Vec(dim):
    seed=123
    window=10
    min_count=0
    sg=1
    workers=5 # see if we can increase it
    epochs = 10

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_set_path = os.path.join(current_dir, 'twitter_dataset', 'train.csv')
    train = pd.read_csv(train_set_path, sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)

    final_list = []
    vect = train['text_cleaned']

    for s in vect:
        final_list.append(s.split(" "))

    w2v_model = Word2Vec(vector_size=dim, seed=seed, window=window,
                        min_count=min_count, sg=sg, workers=workers)

    # vocabulary creation
    # token_word = lista di token (lista di liste)
    token_word = final_list
    w2v_model.build_vocab(token_word, min_count=1)
    total_examples = w2v_model.corpus_count

    w2v_model.train(token_word, total_examples=total_examples, epochs=epochs)

    w2v_path = os.path.join(current_dir, 'twitter_dataset', 'word2vec_models', 'word2vec_model_' + str(dim) + '.pkl')
    with open(w2v_path, "wb") as fOut:
        pickle.dump(w2v_model, fOut)