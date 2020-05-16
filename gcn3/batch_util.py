import numpy as np
from parameters import *
import pickle
from annoy import AnnoyIndex

#If use_images is not true, then annoy file will not be loaded, hence all images will be zeros
if use_images == True:
	#Load annoy file for image representations
	url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))    
	#print(type(url_to_index))
	#print(url_to_index)
	a = AnnoyIndex(image_size)
	a.load(annoy_dir+'annoy.ann') 
	#print(a.get_n_items())
	#print(a.get_item_vector(0), a.get_item_vector(1), a.get_item_vector(2))
	print("annoy file loaded")


#Get VGG representation from image URL
def image_rep(image_url):
	v = np.array([0]*image_size)
	if image_url in ["", 'RANDOM']:
		return v
	try:
		index = url_to_index[image_url.strip()]
		v = np.array(a.get_item_vector(index))
	except:      
		if use_images == True:   
			#Print this if image is not found in annoy file  
			print(image_url + " not found in annoy")
	return v


#Convert data into feed_dict format required
def get_feed_dict(model, batch_text, batch_image, batch_adjacency_mat, batch_target_pos, batch_target_negs):
	feed_dict = {}

	#Fill text context
	batch_text_ph = [[] for i in range(max_context_len)]
	for i in range(batch_size):
		for j in range(max_context_len):  
			batch_text_ph[j].append(batch_text[i][j]) 
	for i in range(max_context_len):
		batch_text_ph[i] = np.array(batch_text_ph[i]) 
	for ph_i, input_i in zip(model.encoder_text_inputs, batch_text_ph):
		feed_dict[ph_i] = input_i  
	#	 
	#Fill image context
	batch_image_ph = [[] for i in range(max_context_len)]
	for i in range(batch_size):  
		for j in range(max_context_len):  
			batch_image_ph[j].append([image_rep(batch_image[i][j][k]) for k in range(num_images_in_context)])
	for i in range(max_context_len):
		batch_image_ph[i] = np.array(batch_image_ph[i])
	for ph_i, input_i in zip(model.encoder_image_inputs, batch_image_ph):
		feed_dict[ph_i] = input_i  
	#
	#Make adjacency matrix	
	feed_dict[model.A] = batch_adjacency_mat
	#
	#Fill positive image
	batch_target_pos = [[image_rep(image) for image in image_set] for image_set in batch_target_pos]
	feed_dict[model.pos_images] = batch_target_pos
	#
	#Fill negative images
	batch_target_negs = [[image_rep(image) for image in image_set] for image_set in batch_target_negs]
	feed_dict[model.negs_images] = batch_target_negs
	
	return feed_dict


	
# def pad_for_batch_size(batch_text, batch_image, batch_target_pos, batch_target_negs):
# 	if(len(batch_text) != batch_size):
# 		#If padding required
# 		pad_size = batch_size - len(batch_target_pos)%batch_size

# 		#Pad text context
# 		empty_data = [start_word_id, end_word_id]+[pad_word_id]*(max_utter_len-2)
# 		empty_data = [empty_data]*max_dialogue_len
# 		empty_data_mat = [empty_data]*pad_size
# 		empty_data_mat = np.array(empty_data_mat)
# 		batch_text = np.vstack((batch_text, empty_data_mat))
# 		#
# 		#Pad image context
# 		empty_data = ""
# 		empty_data = [empty_data]*max_dialogue_len
# 		empty_data_mat = [empty_data]*pad_size
# 		empty_data_mat = np.array(empty_data_mat)
# 		batch_image = np.vstack((batch_image, empty_data_mat))
# 		#
# 		#Pad positive image
# 		empty_data = ["RANDOM"]
# 		empty_data_mat = [empty_data]*pad_size
# 		empty_data_mat = np.array(empty_data_mat)
# 		batch_target_pos = np.vstack((batch_target_pos, empty_data_mat))
# 		#
# 		#Pad negative images
# 		empty_data = ["RANDOM"] * num_neg_images_use
# 		empty_data_mat = [empty_data]*pad_size
# 		empty_data_mat = np.array(empty_data_mat)
# 		batch_target_negs = np.vstack((batch_target_negs, empty_data_mat))
	
# 	return batch_text, batch_image, batch_target_pos, batch_target_negs




def process_batch(model, data_batch):
	#data_batch is a batch_size sized list of (batch_text, batch_image, batch_target_pos, batch_target_negs)
	#get batch_text, batch_image, batch_target_pos and batch_target_negs from data_batch
	batch_text = []
	batch_image = []
	batch_adjacency_mat = []
	batch_target_pos = []
	batch_target_negs = []
	batch_knowledge = []
	batch_num_tuples = []
	for instance in data_batch:
		batch_text.append(instance[0])
		batch_image.append(instance[1])
		batch_adjacency_mat.append(np.array(instance[2].todense()))
		batch_target_pos.append(instance[3])
		batch_target_negs.append(instance[4][:num_neg_images_use])
		batch_knowledge.append(instance[5])
		batch_num_tuples.append(instance[6])
	batch_text = np.array(batch_text)
	batch_image = np.array(batch_image)
	batch_adjacency_mat = np.array(batch_adjacency_mat)
	batch_target_pos = np.array(batch_target_pos) 
	batch_target_negs = np.array(batch_target_negs)
	batch_knowledge = np.array(batch_knowledge)
	batch_num_tuples = np.array(batch_num_tuples)
	#batch_text is [batch_size * max_context_len * max_utter_len]
	#batch_image is [batch_size * max_context_len * num_images_in_context] 
	#batch_adjacency_mat is [batch_size * num_nodes * num_nodes]
	#batch_target_pos is [batch_size * 1]
	#batch_target_negs is [batch_size * num_neg_images]
	#
	#batch_text, batch_image, batch_target_pos, batch_target_negs = pad_for_batch_size(batch_text, batch_image, batch_target_pos, batch_target_negs)
	#print(batch_text.shape, batch_image.shape, batch_target_pos.shape, batch_target_negs.shape)
	#
	feed_dict = get_feed_dict(model, batch_text, batch_image, batch_adjacency_mat, batch_target_pos, batch_target_negs)    
	#batch_text is [batch_size * max_context_len * max_utter_len]
	#batch_image is [batch_size * max_context_len * num_images_in_context * image_size] 
	#batch_adjacency_mat is [batch_size * num_nodes * num_nodes]
	#batch_target_pos is [batch_size * 1 * image_size]
	#batch_target_negs is [batch_size * num_neg_images * image_size]

	return feed_dict




#get rank of positive image among negative images given the similarities
def get_ranks(pos_sim, negs_sim):
	ranks = []
	#print("pos_sim negs_sim shapes ", pos_sim.shape, negs_sim.shape)
	for pos, negs in zip(pos_sim, negs_sim):
		rank = num_neg_images_use+1
		for neg in negs:
			if pos >= neg:
				rank -= 1
		ranks.append(rank)
	return ranks




#calculate recall @ m
def recall(ranks, m):
	correct = 0
	total = 0
	for rank in ranks:
		if rank <= m:
			correct += 1
		total += 1
	return float(correct)/float(total)

