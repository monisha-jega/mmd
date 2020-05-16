from parameters import *
import gensim, numpy, pickle


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

#Load vocab
vocab = pickle.load(open(data_dump_dir+"vocab_id_to_word.pkl","rb"))
vocab_size = len(vocab)
#Load KG
kg = pickle.load(open(data_dump_dir + "image_kg.pkl"))

#Get google embeddings for all vocab words
embeddings_google = []
for i in range(vocab_size):
	word = vocab[i]
	try:
		vector = model[word]
	except:
		vector = [0 for e in range(word_embedding_size)]
	embeddings_google.append(vector)

#dump vocab embeddings
with open(data_dump_dir + 'vocab_embeddings.pkl', 'wb') as f:
	pickle.dump(numpy.array(embeddings_google), f)









