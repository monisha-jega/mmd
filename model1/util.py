import numpy as np
from parameters import *
from model import *
import pickle


def one_hot(feature_index, feature_size):
	#print(feature_index, feature_size)
	v = [0 for i in range(feature_size)]
	v[feature_index] = 1
	return v



kg = pickle.load(open(kg_file, 'rb'))
kg_as_list = kg.items()
total_images = len(kg_as_list)
feature_index_map = pickle.load(open(feature_index_map_file, 'rb'))

def get_image_rep_from_kg(image_url, onehot = False):
	try:
		image_json = kg[image_url]
		#print("ok")
	except:
		# if image_url not in ["RANDOM", ""]:
		# 	print(image_url, "not found in KG")
		if onehot == True:
			return [[0 for x in range(feature_size)] for feature_size in feature_sizes]
		else:
			return [0 for feature_size in feature_sizes]
	
	kg_rep = []
	for feature_name, feature_size in zip(features, feature_sizes):
		feature_index = image_json[feature_name]
		if onehot == True:
			kg_rep.append(one_hot(feature_index, feature_size))
		else:
			kg_rep.append(feature_index)
	return kg_rep



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
		batch_image_reps.append([get_image_rep_from_kg(url, True) for url in data_point[3]])
		batch_target_image_rep.append(get_image_rep_from_kg(data_point[4]))
		batch_neg_images.append([get_image_rep_from_kg(url) for url in data_point[5][:num_neg_images_use]])
		#print("Well", data_point[4])
	#print(batch_target_image_rep)
	feeding_dict = {}
	for j in range(max_context_len):
		feeding_dict[text_inputs_ph[j]] = np.array([batch_text_inputs[i][j] for i in range(batch_size)])
	for j in range(num_features):
	# 	#print(j, " j")
	# 	#print(dialogue_slots_ph[j])
	# 	for i in range(batch_size):
			#print(i, "i")
			#print(len(batch_dialogue_slots[i]))
			#print(len(batch_dialogue_slots[i][j]))
		feeding_dict[dialogue_slots_ph[j]] = np.array([batch_dialogue_slots[i][j] for i in range(batch_size)])

	feeding_dict[last_ds_ph] = np.array(batch_last_ds)
	for j in range(num_images_per_utterance):
		for k in range(num_features):
			feeding_dict[image_reps_ph[j][k]] = np.array([batch_image_reps[i][j][k] for i in range(batch_size)]) 
			# print([batch_image_reps[i][j][k] for i in range(batch_size)])
			# print(j, k)
	for j in range(num_features):
		feeding_dict[target_image_rep_ph[j]] = np.array([batch_target_image_rep[i][j] for i in range(batch_size)]) 
	#print(batch_target_image_rep)
	return feeding_dict, batch_target_image_rep, batch_neg_images



def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def cross_entropy_dist(target, pred):
	#target is num_features
	#pred is num_features * feature_size
	# print("target")
	# print(target)	
	# print("pred")
	# print(pred)
	dist = 0.0
	for f, (target_feature, target_pred) in enumerate(zip(target, pred)):
		#dist += np.dot(target_feature, -np.log(target_pred))
		#print(features[f], target_feature, len(target_pred), target_pred)
		target_pred = softmax(np.array(target_pred))
		#print(f, target_pred[target_feature])
		dist += -np.log(target_pred[target_feature])
	return dist



def rank_images(preds, pos_images, neg_images):
	#preds batch_size * num_features * feature_size
	#pos_images is batch_size * num_features
	#neg_images is batch_size * num_neg_images  * num_features

	ranks = []
	for pred, pos_image, neg_image in zip(preds, pos_images, neg_images)[:]:
		#pred is num_features * feature_size
		#pos_image is num_features
		#neg_image is num_neg_images  * num_features
		pos_dist = cross_entropy_dist(pos_image, pred)
		neg_dists = [cross_entropy_dist(each_image, pred) for each_image in neg_image]
		# print("posdist")
		# print(pos_dist)
		# print("negsdist")
		#print(neg_dists)
		rank = 1
		for neg_dist in neg_dists:
			if pos_dist > neg_dist:
				rank += 1

		ranks.append(rank)
	#print(ranks)
	return ranks


def recall_m(ranks, m):
	correct = 0
	for rank in ranks:
		if rank <= m:
			correct += 1
	return float(correct)/len(ranks)
