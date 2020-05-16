import pickle, json
from util import *
from parameters import *
import nltk, os, sys, random
from collections import *

#Vocab counter
vocab_counter = Counter()
#Dialogue state index map
state_index_map = {}
num_states = 0


#For each dialogue
for e, json_file in enumerate(os.listdir(train_dir)[:]):
	if e%1000 == 0:
		print(e)
	dialogue_json = json.load(open(train_dir + json_file))	
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
		vocab_counter.update(nlg_words)	


		state = utterance_["type"]
		if state == "question":
			if "question-type" in utterance_:
				state = utterance_["question-type"]
			if "question-subtype" in utterance_:
				state = utterance_["question-subtype"]

		if state not in state_index_map:
			state_index_map[state] = num_states
			num_states += 1


#Build Vocabulary after removing rare words
total_freq = sum(vocab_counter.values())
vocab_count = [x for x in vocab_counter.most_common() if x[1] >= vocab_min_freq]	
vocab_dict = {pad_symbol : pad_id, unk_symbol : unk_id}
i = 2
for (word, count) in vocab_count:
	if not word in vocab_dict:
		vocab_dict[word] = i
		i += 1


train_data = []

#For each dialogue
#For each dialogue
for e, json_file in enumerate(os.listdir(train_dir)[:]):
	if e%1000 == 0:
		print(e)
	dialogue_json = json.load(open(train_dir + json_file))	
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
		#print("1")
		#Pad each utterance
		nlg_words = nlg_words[:max_utter_len]
		if len(nlg_words) < max_utter_len:
			nlg_words += [pad_symbol for x in range(max_utter_len - len(nlg_words))]
		list_utterances.append([vocab_dict.get(word, unk_id) for word in nlg_words])	

		if utterance['nlg'] != None:
			current_slots = replace_slots(current_slots, utterance['nlg'].lower())
		list_slots.append(current_slots)
		#print("2")
		state = utterance_["type"]
		if state == "question":
			if "question-type" in utterance_:
				state = utterance_["question-type"]
			if "question-subtype" in utterance_:
				state = utterance_["question-subtype"]
		state_index = state_index_map[state]
		list_states.append(state_index)
		#print("3")
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
			pos_image = random.sample(utterance_images, 1)[0]

			if sample_images == True:
				neg_indices = random.sample(range(total_images), num_neg_images_sample)
				neg_images = [kg_as_list[index][0] for index in neg_indices]
			else:
				#Negative images
				neg_images = utterance_false_images[:num_neg_images_sample]
				if len(neg_images) < num_neg_images_sample:
					pad_length = num_neg_images_sample - len(neg_images)
					neg_images = neg_images + ['RANDOM']*pad_length	
			
			train_instance = [dialogue_context, dialogue_slots, ds, dialogue_images, pos_image, neg_images]
			train_data.append(train_instance)

		if utterance_['type'] == 'question':	
			is_prev_utterance_a_question = True

pickle.dump(vocab_dict, open(vocab_word_to_id_file,'wb'))
pickle.dump(train_data, open(train_data_file, 'wb'))
pickle.dump(state_index_map, open(state_index_map_file, 'wb'))
print("Number of states", len(state_index_map))
print("Train data points", len(train_data))
print("Vocab", len(vocab_dict))

