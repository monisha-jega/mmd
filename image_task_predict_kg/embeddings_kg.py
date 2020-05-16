from parameters_domain_model_full import *
import gensim, pickle
import numpy as np


#To make an embedding matrix for the vocabulary 
#Using pre-trained google vectors


def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


#Load model
model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=True)

#Load KG
kg = pickle.load(open(data_dump_dir + "image_kg_pred.pkl"))

#Get google embeddings for all KG words
embeddings_google = {}


count_found = 0
count_notfound = 0
count_empty = 0
for url, arr in kg.items():
	for val in arr:
		
		try:
			vector = model[val]
			count_found += 1
		except:
			vector = [0 for i in range(word_embedding_size)]
			count_notfound += 1
		embeddings_google[val] =  vector

print(count_found, count_notfound, count_empty)
#dump vocab embeddings
with open(data_dump_dir + 'kg_embeddings.pkl', 'wb') as f:
	pickle.dump(embeddings_google, f)









