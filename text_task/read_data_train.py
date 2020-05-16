from parameters import *
import os, copy, json
import nltk
from read_data_util import *
from collections import *

#Vocab counter initialization
vocab_counter = Counter()

#Files to write training data in readable form
open(data_dump_dir+"train_targets", 'w')
open(data_dump_dir+"train_contexts_text", 'w')
open(data_dump_dir+"train_contexts_image", 'w')


#For each dialogue
for json_file in os.listdir(train_dir)[:]:
	#Load dialogue
	dialogue_json = json.load(open(train_dir+json_file))
	# print(len(dialogue_json))
	
	dialogue_till_now = []
	dialogue_targets = []
	dialogue_contexts_text = []
	dialogue_contexts_image = []

	is_prev_utterance_a_question = False
	#For each utterance in dialogue
	for i, utterance_ in enumerate(dialogue_json):
		print(i)
		utterance = utterance_['utterance'] #Textual part of utterance
		#print(utterance_)
		if utterance is None:
			continue	

		#Tokenize text and update vocabulary
		if utterance['nlg'] is None:
			nlg = ""
		else:
			nlg = utterance['nlg'].strip().encode('utf-8').lower().replace("|","")
		try:
			nlg_words = nltk.word_tokenize(nlg)
			# print(nlg_words)
		except:
			nlg_words = nlg.split(" ")				
		vocab_counter.update(nlg_words)
		

		#Make a training instance out of the dialogue till now, with last utterace as prediction
		if utterance_['speaker'] == "system" and is_prev_utterance_a_question == True and 'nlg' in utterance and utterance['nlg'] not in ["", None]:
			
			padded_clipped_dialogue_instance = pad_or_clip_dialogue(dialogue_till_now)
			assert len(padded_clipped_dialogue_instance) == max_dialogue_len
			
			#Target text
			target_text = utterance['nlg']
			dialogue_targets.append(target_text)
			#
			#Textual context
			context_texts = []
			for x in padded_clipped_dialogue_instance:
				if 'nlg' in x and x['nlg'] not in ["",None]:
					context_texts.append(x['nlg'])
				else:
					context_texts.append("")
			dialogue_contexts_text.append(context_texts)
			#	
			#Image context
			context_images = []
			for x in padded_clipped_dialogue_instance:
				if 'image' in x and x['image'] is not None:
					context_images.append(x['image'])
				else:
					context_images.append("")
			dialogue_contexts_image.append(context_images)


		if utterance_['type'] == 'question':	
			is_prev_utterance_a_question = True
		else:
			is_prev_utterance_a_question = False
		
		#Append context utterance instance
		inst = {}
		if 'images' in utterance:
			inst['images'] = utterance['images']
		if 'nlg' in utterance:
			inst['nlg'] = utterance['nlg']
		dialogue_till_now.append(inst)



	#Write all possible train cases from a single dialogue
	with open(data_dump_dir+"train_targets", 'a') as fp:
		for dialogue_target in dialogue_targets:
			fp.write(dialogue_target.encode('utf-8')+'\n')
	#
	with open(data_dump_dir+"train_contexts_text", 'a') as fp:
		for dialogue_context_text in dialogue_contexts_text:
			dialogue_context_text_write = '|'.join(dialogue_context_text)
			fp.write(dialogue_context_text_write.encode('utf-8')+'\n')	
	#
	with open(data_dump_dir+"train_contexts_image", 'a') as fp:
		for dialogue_context_image in dialogue_contexts_image:
			dialogue_context_image_write = "|".join(dialogue_context_image)
			fp.write(dialogue_context_image_write+'\n')




#Build Vocabulary by filtering out rare words (retain words with freq above cutoff)
vocab_count = [x for x in vocab_counter.most_common() if x[1] >= vocab_freq_cutoff]	
vocab_dict = {unk_word:unk_word_id, start_word: start_word_id, pad_word:pad_word_id, end_word:end_word_id}
i = 4
for (word, count) in vocab_count:
	if not word in vocab_dict:
		vocab_dict[word] = i
		i += 1
		




#Now data is written, take it binarize it and store it
binarized = []
with open(data_dump_dir+"train_targets",) as targetlines, open(data_dump_dir+"train_contexts_text") as textlines, open(data_dump_dir+"train_contexts_image") as imagelines:
		for text_context, image_context, target in zip(textlines, imagelines, targetlines):
		
		#Binarize target
		try:
			utterance_words = nltk.word_tokenize(target)
		except:
			utterance_words = target.split(' ')	
		utterance_words = pad_or_clip_utterance(utterance_words)		
		binarized_target = [vocab_dict.get(word, unk_word_id) for word in utterance_words]
		#
		#Binarize text context
		binarized_context_text = []
		utterances = text_context.lower().strip().split('|')
		for utterance in utterances:
			try:
				utterance_words = nltk.word_tokenize(utterance)
			except:
				utterance_words = utterance.split(' ')
			utterance_words = pad_or_clip_utterance(utterance_words)
			binarized_context_text.append([vocab_dict.get(word, unk_word_id) for word in utterance_words])
		#
		#Binarize image context		
		binarized_context_image = image_context.strip().split('|')

		#Each training instance is appended
		binarized.append([binarized_context_text, binarized_context_image, binarized_target])
print("Length of training data ", len(binarized))


print("pickling")
#Pickle binarized data - training data, vocab and inverted vocab
pickle_func(binarized, data_dump_dir+"train_binarized_data.pkl")	
pickle_func(vocab_dict, data_dump_dir+"vocab_word_to_id.pkl")
inverted_vocab_dict = {word_id : word for word, word_id in vocab_dict.iteritems()}	
pickle_func(inverted_vocab_dict, data_dump_dir+"vocab_id_to_word.pkl")



				
	



