from parameters import *
import os, copy, json, sys, pickle
import nltk
from collections import *


data_type = sys.argv[1]
if data_type == 'val':
	data_dir = val_dir
else:
	data_dir = test_dir


#Load vocab
vocab_dict = pickle.load(open(data_dump_dir+'vocab_dst.pkl', 'rb'))
#Load utterance types and their indices map
state_index_map = json.load(open(data_dump_dir + "state_index_map.json"))

data_readable = []
for json_name in os.listdir(data_dir):
	#Load dialogue
	dialogue = json.load(open(data_dir + json_name))
	#For each utterance in dialogue
	for utterance_ in dialogue:
		utterance = utterance_["utterance"] #Textual part of utterance

		#Tokenize utterance and update vocabulary
		if ("nlg" not in utterance) or (utterance["nlg"] in [None, ""]):
			continue
		nlg = utterance['nlg'].strip().encode('utf-8').lower()	
		try:
			nlg_words = nltk.word_tokenize(nlg)
		except:
			nlg_words = nlg.split(" ")				
			
		#Utterance state index
		state = utterance_["type"]
		if state == "question":
			if "question-type" in utterance_:
				state = utterance_["question-type"]
			if "question-subtype" in utterance_:
				state = utterance_["question-subtype"]
		state_index = state_index_map[state]

		#Append data instance
		data_readable.append([nlg_words, state_index])

#Dump readable val/test data
with open(data_dump_dir + data_type + "_dst_readable.json", 'w') as data_dst_readable:
	json.dump(data_readable, data_dst_readable)	


#Binarize val/test data
data_bin = []
for data_instance in data_readable:
	words, state = data_instance
	#Convert words into vocab indices
 	word_ids = [vocab_dict.get(word, unk_word_id) for word in words]
 	word_ids = word_ids[:MAX_SEQUENCE_LENGTH]
 	if len(word_ids) < MAX_SEQUENCE_LENGTH:
		word_ids += [pad_word_id for p in range(MAX_SEQUENCE_LENGTH - len(word_ids))]
	
	data_bin.append([word_ids, state])
print("Num of " + data_type + " instances", len(data_bin))

#Pickle val/test data
with open(data_dump_dir + data_type + "_dst.pkl", 'wb') as f:
	pickle.dump(data_bin, f)	

