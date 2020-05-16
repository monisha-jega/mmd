from parameters import *
import os, copy, json, sys, random, pickle
import nltk
from read_data_util import *
from collections import *

#For tokenization
nlp = StanfordCoreNLP('http://localhost', port=8000)





data_type = sys.argv[1]
if data_type == 'val':
	data_dir = val_dir
else:
	data_dir = test_dir

#Dependency Parses
depparses = pl.load(open(data_dump_dir + data_type + "_depparses.pkl", 'rb'))

vocab_dict = pickle.load(open(data_dump_dir+'vocab_word_to_id.pkl', 'rb'))



count = 0

#Files to write val/test data in readable form
open(data_dump_dir+data_type+"_contexts_image", 'w')
open(data_dump_dir+data_type+"_contexts_text", 'w')
open(data_dump_dir+data_type+"_targets_pos", 'w')
open(data_dump_dir+data_type+"_targets_negs", 'w')


#For each dialogue
for json_file in os.listdir(data_dir)[:]:
	#Load dialogue
	dialogue_json = json.load(open(data_dir+json_file))
	# print(len(dialogue_json))
	
	dialogue_till_now = []
	dialogue_targets_pos = []
	dialogue_targets_negs = []
	dialogue_contexts_text = []
	dialogue_contexts_image = []

	is_prev_utterance_a_question = False
	#For each utterance in dialogue
	for i, utterance_ in enumerate(dialogue_json):
		#print(i)
		utterance = utterance_['utterance'] #Textual part of utterance
		#print(utterance_)
		if utterance is None:
			continue		

		if utterance['nlg'] is None:
			nlg = ""
		else:
			nlg = utterance['nlg'].strip().encode('utf-8').lower().replace("|","")
		nlg_id = json_file + "_" + str(i)

		#Make an instance out of the dialogue till now, with last utterace as prediction
		if utterance_['speaker'] == "system" and is_prev_utterance_a_question == True and 'images' in utterance and not utterance['images'] == None and len(utterance['images'])>0 and 'false images' in utterance and not utterance['false images'] == None and len(utterance['false images'])>0  and not None in utterance['images'] and not None in utterance['false images']:
			
			padded_clipped_dialogue_instance = pad_or_clip_dialogue(dialogue_till_now)
			assert len(padded_clipped_dialogue_instance) == max_dialogue_len
			
			#Choose a positive target image
			sampled_img_pos = random.sample(utterance['images'],1)		
			dialogue_targets_pos.append(sampled_img_pos[0])
			#
	 		#Get negative images
			if use_random_neg_images == True:
				#Sample random numbers and get their URLs
				neg_indices = random.sample(range(total_images), num_neg_images_sample)
				dialogue_target_negs = [url_to_index_as_list[index][0] for index in neg_indices]
			else:
				#Get negative images from 'False Images' in the dataset itself
				if len(utterance['false images']) < num_neg_images_sample:
					pad_length = num_neg_images_sample - len(utterance['false images'])
					dialogue_target_negs = utterance['false images'] + ['RANDOM']*pad_length				
				elif len(utterance['false images']) > num_neg_images_sample:
					dialogue_target_negs = utterance['false images'][:num_neg_images_sample]
				else:
					dialogue_target_negs = utterance['false images']
			dialogue_targets_negs.append(dialogue_target_negs)		
			#
			#Textual and image context
			context_images = []
			context_texts = []
			for x in padded_clipped_dialogue_instance:
				if 'nlg' in x and x['nlg'] not in ["",None]:
					context_texts.append((nlg_id, x['nlg'][1]))
				else:
					context_texts.append((x['nlg'][0], ""))
				if 'images' in x and x['images'] is not None:
					context_images.append(x['images'])
				else:
					context_images.append([])
			dialogue_contexts_text.append(context_texts)
			dialogue_contexts_image.append(context_images)

			count += 1
			if count % 10000 == 0:
				print(count)
				
		#Is this utterance a question?
		if utterance_['type'] == 'question':	
			is_prev_utterance_a_question = True
		else:
			is_prev_utterance_a_question = False

		#Append context utterance instance
		inst = {}
		if 'images' in utterance:
			inst['images'] = utterance['images']
		if 'nlg' in utterance:
			inst['nlg'] = (nlg_id, nlg)
		dialogue_till_now.append(inst)




	#Write all possible val/test cases from a single dialogue
	with open(data_dump_dir+data_type+"_contexts_text", 'a') as fp:
		for dialogue_context_text in dialogue_contexts_text:
			dialogue_context_text_id_concat = [(text_id + "^^^" + text) for text_id, text in dialogue_context_text]
			dialogue_context_text_write = '|'.join(dialogue_context_text_id_concat)
			fp.write(dialogue_context_text_write.decode('utf-8').encode('utf-8')+'\n')
	#
	with open(data_dump_dir+data_type+"_contexts_image", 'a') as fp:
		for dialogue_context_image in dialogue_contexts_image:
			dialogue_context_image_write = "^^^".join(["|".join(dialogue_context_each_image) for dialogue_context_each_image in dialogue_context_image])
			fp.write(dialogue_context_image_write+'\n')
	#
	with open(data_dump_dir+data_type+"_targets_pos", 'a') as fp:
		for dialogue_target_pos in dialogue_targets_pos:
	 		fp.write(dialogue_target_pos+'\n')
	#
	with open(data_dump_dir+data_type+"_targets_negs", 'a') as fp:
		for dialogue_target_negs in dialogue_targets_negs:
			dialogue_target_negs_write = '|'.join(dialogue_target_negs)
			fp.write(dialogue_target_negs_write+'\n')



#Now data is written, take it binarize it and store it
prev_root = None
binarized = []
with open(data_dump_dir+data_type+"_targets_pos") as targetposlines, open(data_dump_dir+data_type+"_targets_negs",) as targetnegslines, open(data_dump_dir+data_type+"_contexts_text") as textlines, open(data_dump_dir+data_type+"_contexts_image") as imagelines:
	for dno, (text_context, images_context, target_pos, target_negs) in enumerate(zip(textlines, imagelines, targetposlines, targetnegslines)):
		
		#Binarize pos target
		binarized_target_pos = [target_pos.strip()]
		#
		#Binarize negative images
		binarized_target_negs = target_negs.strip().split('|')
		#

		#Adjacency matrix
		#Make according to dependency parse and entity matching
		adjacency = np.zeros((num_nodes, num_nodes))

		#
		#Binarize image context	
		actual_images_array = []	
		binarized_context_images = images_context.strip().split('^^^')
		#print(images_context, "concn")
		for ui in range(len(binarized_context_images)):
			if binarized_context_images[ui] == "":
				utterance_images  = []
			else:
				utterance_images = binarized_context_images[ui].strip().split("|")
			actual_num_images_in_context = len(utterance_images)
			actual_images_array.append(actual_num_images_in_context)
			binarized_context_images[ui] = utterance_images
			if actual_num_images_in_context < num_images_in_context:
				binarized_context_images[ui] += ["" for l in range(num_images_in_context - actual_num_images_in_context)]
		#print(np.array(binarized_context_images).shape)

		#Binarize text context
		binarized_context_text = []
		utterances = text_context.lower().strip().split('|')
		tax_words = {}
		for uno, utterance_and_id in enumerate(utterances):
			nlg_id, utterance = utterance_and_id.split("^^^")
			actual_num_images_in_context = actual_images_array[uno]
			parse = depparses[nlg_id]

			if len(parse) == 0:
				utterance_words = pad_or_clip_utterance([])
				binarized_context_text.append([vocab_dict.get(word, unk_word_id) for word in utterance_words])
				#Attach images to previous root
				if prev_root != None:
					for img in range(actual_num_images_in_context):
						adjacency[uno*(max_utter_len+num_images_in_context)+max_utter_len + img][(uno-1)*(max_utter_len+num_images_in_context) + prev_root] = 1
						#print(uno, img, "prev" , uno-1, root.i)
				prev_root = None
				continue
			try:
				utterance_words = nlp.word_tokenize(utterance)
			except:
				utterance_words = utterance.split(' ')

			#DEP PARSE
			root = parse[0]-1
			#Add connections from subject of one utterance to the next
			if consecutive_subject_connections == True and uno > 0 and prev_root != None and root != None:
				adjacency[(uno-1)*(max_utter_len+num_images_in_context)+prev_root][uno*(max_utter_len+num_images_in_context)+root] = 1
			prev_root = root
			
			add_dep_edges(uno, parse, adjacency)
			#----------------------------------
			#ENTITY MATCHING AND ADDING EDGES
			#----------------------------------
			#print(utterance_words)
			found_at_least_one_tax_word = True
			for wno, word in enumerate(utterance_words[:max_utter_len]):
				found_at_least_one_tax_word = False
				if word in tax_list:
					#print(uno, wno, word)
					#if word is in taxonomy, connect to each image
					for img in range(actual_num_images_in_context):
						adjacency[uno*(max_utter_len+num_images_in_context)+max_utter_len + img][uno*(max_utter_len+num_images_in_context) + wno] = 1
						#print(uno, wno, word, img)
						#print(uno*(max_utter_len+num_images_in_context)+max_utter_len + img, uno*(max_utter_len+num_images_in_context) + wno)
					found_at_least_one_tax_word = True
					if word in tax_words:
						for prev_uno, prev_wno in tax_words[word]:
							adjacency[prev_uno*(max_utter_len+num_images_in_context)+prev_wno][uno*(max_utter_len+num_images_in_context)+wno] = 1
							#print(prev_uno, prev_wno, "entity to", word, uno, wno)
						tax_words[word].append((uno, wno))						
					else:
						tax_words[word] = [(uno, wno)]

			#----------------------------------
			#Need to connect image to subject of each sentence if required
			if found_at_least_one_tax_word == False:				
				for img in range(actual_num_images_in_context):
					adjacency[uno*(max_utter_len+num_images_in_context)+max_utter_len + img][uno*(max_utter_len+num_images_in_context) + (root)] = 1
					#print(uno, img, sent.root.i)
					#print(uno*(max_utter_len+num_images_in_context)+max_utter_len + img, uno*(max_utter_len+num_images_in_context) + (sent.root.i))
					#print("notax")
			utterance_words = pad_or_clip_utterance(utterance_words)
			binarized_context_text.append([vocab_dict.get(word, unk_word_id) for word in utterance_words])
		
		if dno % 10000 == 0:
			print(dno)

		sparse_adjacency = sparse.csr_matrix(adjacency)
		
		#Each val or test instance is appended
		binarized.append([binarized_context_text, binarized_context_images, sparse_adjacency, binarized_target_pos, binarized_target_negs])
print("Length of " + data_type + " data ", len(binarized))


print("pickling")
#Pickle binarized data
pickle_func(binarized, data_dump_dir+data_type+"_binarized_data.pkl")
