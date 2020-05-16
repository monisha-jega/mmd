from parameters import *
import gensim, numpy, pickle


#To make an embedding matrix for the vocabulary 
#Using pre-trained google vectors


#Load model
model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=True)
#Load vocab
vocab = pickle.load(open(vocab_word_to_id_file,"rb"))
inverted_vocab = {id_ : word for word, id_ in vocab.items()}
vocab_size = len(vocab)
print(vocab_size)

#Get google embeddings for all vocab words
embeddings_google = []
for i in range(vocab_size):
	word = inverted_vocab[i]
	try:
		vector = model[word]
	except:
		vector = [0 for e in range(embedding_size)]
	embeddings_google.append(vector)

#dump vocab embeddings
with open(vocab_embeddings_file, 'wb') as f:
	pickle.dump(numpy.array(embeddings_google), f)









