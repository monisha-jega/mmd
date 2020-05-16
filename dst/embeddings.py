from parameters import *
import gensim, numpy, pickle


#To make an embedding matrix for the vocabulary using pre-trained google vectors


#Load pretrained model
model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=True)
#Load vocab
vocab = pickle.load(open(data_dump_dir+"vocab_dst.pkl","rb"))

#Get google embeddings for all vocab words
embeddings_google = []
for word, index, in vocab.items()[:]:
	try:
		vector = model[word]
	except:
		vector = [0 for i in range(word_embedding_size)]
	embeddings_google.append(vector)

#dump vocab embeddings
with open(data_dump_dir + 'vocab_embeddings_dst.pkl', 'wb') as f:
	pickle.dump(numpy.array(embeddings_google), f)









