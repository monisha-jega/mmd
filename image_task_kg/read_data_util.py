from parameters import *
import pickle




#Convert utterance with multiple images into multiple utterances(in context)
def rollout_dialogue(dialogue_instance):
	rolledout_dialogue_instance = []
	for utterance in dialogue_instance:
		if 'nlg' in utterance and utterance['nlg'] is not None:
			rolledout_dialogue_instance.append({"nlg":utterance['nlg'] , "images":None})
		if 'images' in utterance and utterance['images'] is not None and len(utterance['images']) > 0:
			for image in utterance['images']:
				rolledout_dialogue_instance.append({"image":image,"nlg":None})

	return rolledout_dialogue_instance



#Pad or clip contexts to max_dialogue_len
def pad_or_clip_dialogue(dialogue_instance):
	dialogue_instance = rollout_dialogue(dialogue_instance)

	if len(dialogue_instance) > max_dialogue_len:
		clipped_dialogue_instance = dialogue_instance[-(max_dialogue_len):]
		return clipped_dialogue_instance

	elif len(dialogue_instance) < max_dialogue_len:			
		pad_length = max_dialogue_len - len(dialogue_instance)
		padded_dialogue_instance = [{'image':"", 'nlg':""}]*pad_length
		padded_dialogue_instance.extend(dialogue_instance)
		return padded_dialogue_instance	

	else:
		return dialogue_instance


#Pad or clip utterances to max_utter_len
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
	pickle.dump(data, f)
	f.close()