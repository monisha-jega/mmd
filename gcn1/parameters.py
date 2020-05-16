# train_dir = "/scratch/scratch2/monishaj/dataset/v1/valid/"
# val_dir = "/scratch/scratch2/monishaj/dataset/v1/valid/"
# test_dir = "/scratch/scratch2/monishaj/dataset/v1/test/"
# data_dump_dir = "/scratch/scratch2/monishaj/image_data_dump_SN/"
# model_dump_dir = "/scratch/scratch2/monishaj/image_model_dump_SN/"
# #Must contain ImageUrlToIndex.pkl and annoy.ann
# annoy_dir = '/scratch/scratch2/monishaj/image_annoy_index/'
# #Word embeddings file  
# embed_file = '../../GoogleNews-vectors-negative300.bin'

train_dir = "../../dataset/v1/train/"
val_dir = "../../dataset/v1/valid/"
test_dir = "../../dataset/v1/test/"
data_dump_dir = "data_dump/"
model_dump_dir = "model_dump/"
#Must contain ImageUrlToIndex.pkl and annoy.ann
annoy_dir = '../../raw_catalog/image_annoy_index/'
#Word embeddings file  
embed_file = '../../GoogleNews-vectors-negative300.bin'


start_word = "</s>"
start_word_id = 0
end_word = "</e>"
end_word_id = 1
pad_word = "<pad>"
pad_word_id = 2
unk_word = "<unk>"
unk_word_id = 3


max_dialogue_len = max_context_len = 20	 	#Time steps for dialogue (Number of utterances in a dialogue)
#max_dialogue_len is used while preprocessing data, while max_context_len is used during training
max_utter_len = 30  	 					#Time steps for utterance (Number of words in a utterance)
num_neg_images_sample = 100					#Number of negative images to train against
num_neg_images_use = 5                      #Number of negative images to test against
num_images_in_context = 5

num_nodes = max_context_len * (1 + num_images_in_context)
word_embedding_size = 300
image_size = 4096
image_embedding_size = 512
cell_state_size = 512
gcn_layer1_out_size = 500

batch_size = 10		    		#best value - 64
vocab_freq_cutoff = 4				#best value - 4
learning_rate=0.0004    			#best value - 0.0004
max_gradient_norm = 0.1 			#best value - 0.1
epochs = 10							#best value - Early stopping

use_random_neg_images = False       #If True, random images are sampled for negative images used in read_data_*.py)
use_images = False					#If False, will use 0s for image instead of loading from annoy file
restore_trained = False 			#If True, will restore latest model from checkpoint