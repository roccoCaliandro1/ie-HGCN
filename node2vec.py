from sentence_transformers import SentenceTransformer
import pandas as pd

# aprire con pandas
# Antonio TRAIN train = COPIA QUI:
# Rocco TEST: test = pd.read_csv('/content/drive/MyDrive/BigData Notebooks/dataset/test.csv', sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)
test = pd.read_csv('./OneDrive/preprocess/twitter/cleaned/small/test.txt', sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)
print(test)
#train.iloc['text_cleaned']
model = SentenceTransformer('digio/Twitter4SSE')
embeddings = model.encode(test['text_cleaned'])
print(embeddings)

# save into directory
import pickle
path = './twitter_dataset/'
with open(path+'test_embeddings.pkl', "wb") as fOut:
    pickle.dump({'userId': test['id'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

