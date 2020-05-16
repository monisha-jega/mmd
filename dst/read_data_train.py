from parameters import *
import os, json, sys, pickle
from collections import *
import nltk

#Vocab counter initialization
vocab_counter = Counter()

#Dialogue states initialization - map from state name to state index, starting from 0
state_index_map = {}
num_states = 0

train_readable = []
for json_name in os.listdir(train_dir)[:]:
	#open dialogue
	dialogue = json.load(open(train_dir + json_name))
	#For each utterance in dialogue
	for utterance_ in dialogue:
		utterance = utterance_["utterance"] #Textua part of utterance

		#Tokenize text and update vocabulary
		if ("nlg" not in utterance) or (utterance["nlg"] in [None, ""]):
			continue
		nlg = utterance['nlg'].strip().encode('utf-8').lower()	
		try:
			nlg_words = nltk.word_tokenize(nlg)
		except:
			nlg_words = nlg.split(" ")				
		vocab_counter.update(nlg_words)
		
		#Utterance type
		state = utterance_["type"]
		if state == "question":
			if "question-type" in utterance_:
				state = utterance_["question-type"]
			if "question-subtype" in utterance_:
				state = utterance_["question-subtype"]
		#Add utterance type to utterance type dictionary
		if state not in state_index_map:
			state_index_map[state] = num_states
			state_index = num_states
			num_states += 1
		else:
			state_index = state_index_map[state]
		
		#Append training instance
		train_readable.append([nlg_words, state_index])

#Dump readable training data
with open(data_dump_dir + "train_dst_readable.json", 'w') as train_dst_readable:
	json.dump(train_readable, train_dst_readable)	
#Dump utterance types
json.dump(state_index_map, open(data_dump_dir + "state_index_map.json", 'w'))
print("Num types", num_states)



#Make Vocab Dict considering only words with count > vocab_freq_cutoff
vocab_count = [x for x in vocab_counter.most_common() if x[1] >= vocab_freq_cutoff]	
vocab_dict = {unk_word:unk_word_id, start_word: start_word_id, pad_word:pad_word_id, end_word:end_word_id}
i = 4
for (word, count) in vocab_counter.items():
	if not word in vocab_dict:
		vocab_dict[word] = i
		i += 1
pickle_func(vocab_dict, data_dump_dir+"vocab_dst.pkl")



#Binarize the training data
train_bin = []
for train_instance in train_readable:
	words, state = train_instance
	#Convert words into vocab indices
	word_ids = [vocab_dict.get(word, unk_word_id) for word in words]
	word_ids = word_ids[:MAX_SEQUENCE_LENGTH]
	if len(word_ids) < MAX_SEQUENCE_LENGTH:
		word_ids += [pad_word_id for p in range(MAX_SEQUENCE_LENGTH - len(word_ids))]

	train_bin.append([word_ids, state])
print("Num of training instances ", len(train_bin))


#Pickle training data
with open(data_dump_dir + "train_dst.pkl", 'wb') as f:
	pickle.dump(train_bin, f)	

