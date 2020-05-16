import tensorflow as tf


#data_dump_dir = "data_dump/"
#model_dump_dir = "model_dump/"
data_dump_dir = "/scratch/scratch2/monishaj/image_data_dump/"
model_dump_dir = "/scratch/scratch2/monishaj/image_model_dump_predict_kg/"
#Must contain ImageUrlToIndex.pkl and annoy.ann
annoy_dir = '/scratch/scratch2/monishaj/image_annoy_index/'
#Word embeddings file  
embed_file = '../GoogleNews-vectors-negative300.bin'

word_embedding_size = 300
image_size = 4096
batch_size = 5		    			#best value - 64
learning_rate=1    				#best value - 0.0004
epochs = 10							#best value - Early stopping

use_images = False					#If 0, will use 0s for image instead of loading from annoy file
restore_trained = False 			#If true, will restore latest model from checkpoint


