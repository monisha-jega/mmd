import numpy as np
from parameters import *
from model import *
import pickle
from annoy import AnnoyIndex

def one_hot(feature_index, feature_size):
	#print(feature_index, feature_size)
	v = [0 for i in range(feature_size)]
	v[feature_index] = 1
	return v


if use_images == True:
	#Load annoy file for image representations
	url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))
	url_to_index_as_list = url_to_index.items()
	total_images = len(url_to_index)
	a = AnnoyIndex(image_size)
	a.load(annoy_dir+'annoy.ann') 
	print("annoy file loaded")

def image_rep(image_url):
	v = np.array([0]*image_size)
	if image_url in ["", 'RANDOM']:
		return v
	try:
		index = url_to_index[image_url.strip()]
		#print("got_index ", index)    
		v = np.array(a.get_item_vector(index))
		#print("sucess")
	except:      
		if use_images == 1:     
			print(image_url + " exception loading from annoy")
	return v

feature_index_map = pickle.load(open(feature_index_map_file, 'rb'))

def fill_slots(utterance):
	slots = []
	for feature_name, feature_size in zip(features, feature_sizes):
		slots.append(-1)
		for feature_val, index in feature_index_map[feature_name].items():
			if feature_val in utterance.lower():
				slots[-1] = index
				break
	return slots


def replace_slots(current_slots, utterance):
	# new_slots = list(current_slots)
	for f, feature_name in enumerate(features):
		for feature_val, index in feature_index_map[feature_name].items():
			if feature_val in utterance.lower():
				current_slots[f] = index
				break
	#return new_slots
	return current_slots


def onehot_slots(slots):
	one_hot_slots = [one_hot(slot, feature_size) for slot, feature_size in zip(slots, feature_sizes)]
	return one_hot_slots



def process_batch(data):

	batch_text_inputs = []
	batch_dialogue_slots = []
	batch_last_ds = []
	batch_image_reps = []    
	batch_target_image_rep = []
	batch_neg_images = []
	
	for data_point in data:
		batch_text_inputs.append(data_point[0]) 
		batch_dialogue_slots.append(onehot_slots(data_point[1]))
		batch_last_ds.append(one_hot(data_point[2], num_ds))
		batch_image_reps.append([image_rep(url) for url in data_point[3]])
		batch_target_image_rep.append(image_rep(data_point[4]))
		batch_neg_images.append([image_rep(url) for url in data_point[5][:num_neg_images_use]])

	feeding_dict = {}

	for j in range(max_context_len):		
		feeding_dict[text_inputs_ph[j]] = np.array([batch_text_inputs[i][j] for i in range(batch_size)])
	
	for j in range(num_features):	
		feeding_dict[dialogue_slots_ph[j]] = np.array([batch_dialogue_slots[i][j] for i in range(batch_size)])

	feeding_dict[last_ds_ph] = np.array(batch_last_ds)

	#Context
	for j in range(num_images_per_utterance):
		feeding_dict[image_inputs[j]] = np.array([batch_image_reps[i][j] for i in range(batch_size)]) 
					
	#Fill positive image
	feeding_dict[pos_images] = np.reshape(np.array(batch_target_image_rep), (batch_size, 1, image_size))

	#Fill negative images
	#batch_target_negs = [[image for image in image_set] for image_set in batch_neg_images]
	feeding_dict[negs_images] = batch_neg_images

	return feeding_dict


def get_ranks(pos_sim, negs_sim):
	ranks = []
	#print("pos_sim negs_sim shapes ", pos_sim.shape, negs_sim.shape)
	for pos, negs in zip(pos_sim, negs_sim):
		rank = 5+1
		for neg in negs:
			if pos >= neg:
				rank -= 1
		ranks.append(rank)
	return ranks





def recall(ranks, m):
	correct = 0
	total = 0

	for rank in ranks:
		if rank <= m:
			correct += 1
		total += 1
	return float(correct)/float(total)

