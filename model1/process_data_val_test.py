import pickle, os, sys, json, random
from util import *
from parameters import *
import nltk
from collections import *


if sys.argv[1] == "val":
	data_dir = val_dir
	data_file = val_data_file
else:
	data_dir = test_dir
	data_file = test_data_file

vocab_dict = pickle.load(open(vocab_word_to_id_file, 'rb'))
state_index_map = pickle.load(open(state_index_map_file, 'rb'))


data = []

#For each dialogue
for json_file in os.listdir(data_dir)[:]:

	dialogue_json = json.load(open(data_dir + json_file))	
	is_prev_utterance_a_question = False
	list_utterances = []
	list_states = []
	list_slots = []
	list_images = []
	current_slots = [-1 for f in range(len(features))]
	for i, utterance_ in enumerate(dialogue_json):
		
		utterance = utterance_['utterance']
		if utterance is None:
			continue	
		#Tokenize and update vocabulary
		if utterance['nlg'] is None:
			nlg = ""
		else:
			nlg = utterance['nlg'].strip().encode('utf-8').lower().replace("|","")	
		try:
			nlg_words = nltk.word_tokenize(nlg)
		except:
			nlg_words = nlg.split(" ")	
		#Pad each utterance
		nlg_words = nlg_words[:max_utter_len]
		if len(nlg_words) < max_utter_len:
			nlg_words += [pad_symbol for x in range(max_utter_len - len(nlg_words))]
		list_utterances.append([vocab_dict.get(word, unk_id) for word in nlg_words])	

		if utterance['nlg'] != None:
			current_slots = replace_slots(current_slots, utterance['nlg'].lower())
		list_slots.append(current_slots)

		state = utterance_["type"]
		if state == "question":
			if "question-type" in utterance_:
				state = utterance_["question-type"]
			if "question-subtype" in utterance_:
				state = utterance_["question-subtype"]
		state_index = state_index_map[state]
		list_states.append(state_index)

		utterance_images, utterance_false_images = [], []
		
		if 'images' in utterance and utterance['images'] not in [[], None, ""]:
			utterance_images = [url for url in utterance['images'] if url is not None]
			context_images = utterance_images[:num_images_per_utterance]
			if len(context_images) < num_images_per_utterance:
				context_images += ["RANDOM" for y in range(num_images_per_utterance - len(context_images))]
			list_images.append(context_images)
			#print(utterance['images'])
		if 'false images' in utterance and utterance['false images'] not in [[], None, ""]:
			utterance_false_images = [url for url in utterance['false images'] if url is not None]
			#print(utterance['false images'] )

		#print(i > 0, utterance_['speaker'] == "system", is_prev_utterance_a_question == True , len(utterance_images) > 0, len(utterance_false_images) > 0)

		#Make a training instance out of the dialogue till now, with last utterace as prediction
		if i > 0 and utterance_['speaker'] == "system" and is_prev_utterance_a_question == True and len(utterance_images) > 0 and len(utterance_false_images) > 0:
			
			#print("4")
			#Textual context
			dialogue_context = list_utterances[-max_context_len:]
			if len(dialogue_context) < max_context_len:
				dialogue_context = [[pad_id for x in range(max_utter_len)] for y in range(max_context_len - len(dialogue_context))] + dialogue_context
			
			#Dialogue slots
			dialogue_slots = list_slots[-2]

			#State
			ds = list_states[-2]
			#print("5")
			#Image Context
			try:
				dialogue_images = list_images[-2]
			except:
				dialogue_images = ["RANDOM" for xx in range(num_images_per_utterance)]
			
			#Positive images
			pos_image = random.sample(utterance_images,1)[0]

			if sample_images == True:
				neg_indices = random.sample(range(total_images), num_neg_images_sample)
				neg_images = [kg_as_list[index][0] for index in neg_indices]
			else:
				#Negative images
				neg_images = utterance_false_images[:num_neg_images_sample]
				if len(neg_images) < num_neg_images_sample:
					pad_length = num_neg_images_sample - len(neg_images)
					neg_images = neg_images + ['RANDOM']*pad_length

			#print("6")
			data_instance = [dialogue_context, dialogue_slots, ds, dialogue_images, pos_image, neg_images]
			data.append(data_instance)
			#print(pos_image, neg_images)

		if utterance_['type'] == 'question':	
			is_prev_utterance_a_question = True

pickle.dump(data, open(data_file, 'wb'))
print(len(data))