from parameters import *
import os, copy, json
import nltk, random
from read_data_util import *
from collections import *

#Vocab counter
vocab_counter = Counter()

open(data_dump_dir+"train_contexts_image", 'w')
open(data_dump_dir+"train_contexts_text", 'w')
open(data_dump_dir+"train_targets_pos", 'w')
open(data_dump_dir+"train_targets_negs", 'w')

#For each dialogue
for json_file in os.listdir(train_dir)[:]:

	dialogue_json = json.load(open(train_dir+json_file))
	# print(len(dialogue_json))
	
	dialogue_till_now = []
	dialogue_targets_pos = []
	dialogue_targets_negs = []
	dialogue_contexts_text = []
	dialogue_contexts_image = []

	is_prev_utterance_a_question = False
	#For each utterance in dialogue
	i = 0
	for utterance_ in dialogue_json:
		print(i)
		i += 1
		utterance = utterance_['utterance']
		#print(utterance_)
		if utterance is None:
			continue	

		#Tokenize and update vocabulary
		if utterance['nlg'] is None:
			nlg = ""
		else:
			nlg = utterance['nlg'].strip().encode('utf-8')			
		nlg = nlg.lower().replace("|","")
		try:
			nlg_words = nltk.word_tokenize(nlg)
			# print(nlg_words)
		except:
			nlg_words = nlg.split(" ")				
		vocab_counter.update(nlg_words)
		#

		

		#Make a training instance out of the dialogue till now, with last utterace as prediction
		if utterance_['speaker'] == "system" and is_prev_utterance_a_question == True and 'images' in utterance and not utterance['images'] == None and len(utterance['images'])>0 and 'false images' in utterance and not utterance['false images'] == None and len(utterance['false images'])>0 and not None in utterance['images'] and not None in utterance['false images']:
			padded_clipped_dialogue_instance = pad_or_clip_dialogue(dialogue_till_now)
			assert len(padded_clipped_dialogue_instance) == max_dialogue_len
			
			sampled_img_pos = random.sample(utterance['images'],1)		
			dialogue_targets_pos.append(sampled_img_pos[0])

			if len(utterance['false images']) < num_neg_images:
				pad_length = num_neg_images - len(utterance['false images'])
				dialogue_target_negs = utterance['false images'] + ['RANDOM']*pad_length				
			elif len(utterance['false images']) > num_neg_images:
				dialogue_target_negs = utterance['false images'][:num_neg_images]
			else:
				dialogue_target_negs = utterance['false images']
			dialogue_targets_negs.append(dialogue_target_negs)



			#Textual context
			context_texts = []
			for x in padded_clipped_dialogue_instance:
				if 'nlg' in x and x['nlg'] not in ["",None]:
					context_texts.append(x['nlg'])
				else:
					context_texts.append("")
			dialogue_contexts_text.append(context_texts)
			
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
		if 'question-type' in utterance_:
			last_question_type = utterance_['question-type']
		else:
			last_question_type = None
		if 'images' in utterance and 'nlg' in utterance:		
			dialogue_till_now.append({'images': utterance['images'], 'nlg':utterance['nlg']})			
		elif 'nlg' in utterance:
			dialogue_till_now.append({'nlg':utterance['nlg']})
		elif 'images' in utterance:
			dialogue_till_now.append({'images': utterance['images']})





	#Write all possible train - test cases from a single dialogue
	with open(data_dump_dir+"train_contexts_text", 'a') as fp:
		for dialogue_context_text in dialogue_contexts_text:
			dialogue_context_text_write = '|'.join(dialogue_context_text)
			fp.write(dialogue_context_text_write.encode('utf-8')+'\n')	

	with open(data_dump_dir+"train_contexts_image", 'a') as fp:
		for dialogue_context_image in dialogue_contexts_image:
			dialogue_context_image_write = "|".join(dialogue_context_image)
			fp.write(dialogue_context_image_write+'\n')


	with open(data_dump_dir+"train_targets_pos", 'a') as fp:
		for dialogue_target_pos in dialogue_targets_pos:
	 		fp.write(dialogue_target_pos+'\n')

	with open(data_dump_dir+"train_targets_negs", 'a') as fp:
		for dialogue_target_negs in dialogue_targets_negs:
			dialogue_target_negs_write = '|'.join(dialogue_target_negs)
			fp.write(dialogue_target_negs_write+'\n')










#Build Vocabulary
total_freq = sum(vocab_counter.values())
vocab_count = [x for x in vocab_counter.most_common() if x[1] >= vocab_freq_cutoff]
	
vocab_dict = {unk_word:unk_word_id, start_word: start_word_id, pad_word:pad_word_id, end_word:end_word_id}
i = 4
for (word, count) in vocab_count:
	if not word in vocab_dict:
		vocab_dict[word] = i
		i += 1
		








binarized = []
with open(data_dump_dir+"train_targets_pos",) as targetposlines, open(data_dump_dir+"train_targets_negs",) as targetnegslines, open(data_dump_dir+"train_contexts_text") as textlines, open(data_dump_dir+"train_contexts_image") as imagelines:
	num_instances = 0
	
	for text_context, image_context, target_pos, target_negs in zip(textlines, imagelines, targetposlines, targetnegslines):
		text_context = text_context.lower().strip()
		image_context = image_context.strip()
		
		#Binarize target
		binarized_target_pos = [target_pos.strip()]
		binarized_target_negs = target_negs.strip().split('|')

		#Binarize text context
		binarized_context_text = []
		utterances = text_context.split('|')
		for utterance in utterances:
			try:
				utterance_words = nltk.word_tokenize(utterance)
			except:
				utterance_words = utterance.split(' ')
			utterance_words = pad_or_clip_utterance(utterance_words)
			binarized_context_text.append([vocab_dict.get(word, unk_word_id) for word in utterance_words])

		#Binarize image context		
		binarized_context_image = image_context.split('|')

		binarized.append([binarized_context_text, binarized_context_image, binarized_target_pos, binarized_target_negs])


#Pickle binarized data
pickle_func(binarized, data_dump_dir+"train_binarized_data.pkl")	

pickle_func(vocab_dict, data_dump_dir+"vocab_word_to_id.pkl")
inverted_vocab_dict = {word_id : word for word, word_id in vocab_dict.iteritems()}	
pickle_func(inverted_vocab_dict, data_dump_dir+"vocab_id_to_word.pkl")



				
	



