import tensorflow as tf

train_dir = "/scratch/scratch2/monishaj/dataset/v1/valid/"
val_dir = "/scratch/scratch2/monishaj/dataset/v1/valid/"
test_dir = "/scratch/scratch2/monishaj/dataset/v1/test/"
data_dump_dir = "/scratch/scratch2/monishaj/image_data_dump/"
model_dump_dir = "/scratch/scratch2/monishaj/image_model_dump_kg/"
#Must contain ImageUrlToIndex.pkl and annoy.ann
annoy_dir = '/scratch/scratch2/monishaj/image_annoy_index/'
#Word embeddings file  
embed_file = '../GoogleNews-vectors-negative300.bin'

start_word = "</s>"
start_word_id = 0
end_word = "</e>"
end_word_id = 1
pad_word = "<pad>"
pad_word_id = 20
unk_word = "<unk>"
unk_word_id = 3

#max_dialogue_len is used while preprocessing data, while max_context_len is used during training
max_dialogue_len = max_context_len = 20	 	#Time steps for dialogue (Number of utterances in a dialogue)
max_utter_len = 30  	 					#Time steps for utterance (Number of words in a utterance)
num_neg_images = 5							#Number of wrong images

word_embedding_size = 300
image_vgg_size = 4096
image_kg_size = 300 + 300 + 4
image_size = image_vgg_size + image_kg_size
image_embedding_size = 512
cell_state_size = 512

batch_size = 64		    			#best value - 64
vocab_freq_cutoff = 4				#best value - 4
learning_rate=0.0004    			#best value - 0.0004
max_gradient_norm = 0.1 			#best value - 0.1
epochs = 10							#best value - Early stopping

#Value of m for recall @ m evaluation
m_for_recall = [1, 2, 3, 4]

use_images = True						#If 0, will use 0s for image instead of loading from annoy file
restore_trained = False 			#If true, will restore latest model from checkpoint
use_kg = True