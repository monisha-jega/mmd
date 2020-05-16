from parameters import *
import numpy as np
import cPickle as pl
from scipy import sparse

if use_random_neg_images == True:
	url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))
	url_to_index_as_list = url_to_index.items()
	total_images = len(url_to_index)


	
#Convert utterance with multiple images into multiple utterances (in context)
#Each utterance is either text or image
def rollout_dialogue(dialogue_instance):
	rolledout_dialogue_instance = []
	for utterance in dialogue_instance:
		rolledout_utterance = {}
		if 'nlg' in utterance and utterance['nlg'] not in ["", None]:
			rolledout_utterance['nlg'] = utterance['nlg']
		else:
			rolledout_utterance['nlg'] = ""
		if 'images' in utterance and utterance['images'] is not None and len(utterance['images']) > 0:
			rolledout_utterance['images'] = [img for img in utterance['images'] if img is not None][:num_images_in_context]
		else:
			rolledout_utterance['images'] = []
		rolledout_dialogue_instance.append(rolledout_utterance)
	return rolledout_dialogue_instance



#Pad or clip contexts to max_dialogue_len
def pad_or_clip_dialogue(dialogue_instance):
	dialogue_instance = rollout_dialogue(dialogue_instance)

	if len(dialogue_instance) > max_dialogue_len:
		clipped_dialogue_instance = dialogue_instance[-(max_dialogue_len):]
		return clipped_dialogue_instance
	elif len(dialogue_instance) < max_dialogue_len:			
		pad_length = max_dialogue_len - len(dialogue_instance)
		padded_dialogue_instance = [{'images': [], 'nlg':""}] * pad_length
		padded_dialogue_instance.extend(dialogue_instance)
		return padded_dialogue_instance	
	else:
		return dialogue_instance


#Pad or clip text utterances to max_utter_len, also add start and end words
def pad_or_clip_utterance(utterance):

	if len(utterance) > max_utter_len-2:
		utterance = utterance[:(max_utter_len-2)]
		utterance.append(end_word)
		utterance.insert(0, start_word)
	elif len(utterance) < max_utter_len-2:
		pad_length = max_utter_len - 2 - len(utterance)
		utterance.append(end_word)
		utterance.insert(0, start_word)
		utterance = utterance + [pad_word]*pad_length
	else:
		utterance.append(end_word)
		utterance.insert(0, start_word)	

	return utterance



#Pickle utility
def pickle_func(data, filename):
	f = open(filename, 'wb')
	pl.dump(data, f)
	f.close()