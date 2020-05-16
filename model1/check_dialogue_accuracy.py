import pickle, json, os, random
from parameters import *
from util import *


def correctness(image, slots):
	image_rep = get_image_rep_from_kg(image)
	print(image_rep, slots)
	correct = []
	for imgf, diaf in zip(image_rep, slots):
		if imgf == diaf:
			correct.append(1)
		else:
			correct.append(0)
	return correct


def replace_slots(current_slots, utterance):
	# new_slots = list(current_slots)
	for f, feature_name in enumerate(features):
		for feature_val, index in feature_index_map[feature_name].items():
			if feature_val in utterance.lower() and feature_val != "" and len(feature_val) > 2:
				print(f, feature_val, utterance)
				current_slots[f] = index
				break
	#return new_slots
	return current_slots



correctness_list = []

#For each dialogue
for e, json_file in enumerate(os.listdir(val_dir)[:1]):
	if e%1000 == 0:
		print(e)
	dialogue_json = json.load(open(val_dir + json_file))	
	is_prev_utterance_a_question = False
		
	list_slots = []
	list_images = []
	current_slots = [-1 for f in range(len(features))]

	for i, utterance_ in enumerate(dialogue_json):
		
		utterance = utterance_['utterance']
		if utterance is None:
			continue			
		if utterance['nlg'] != None:
			current_slots = replace_slots(current_slots, utterance['nlg'].lower())
		list_slots.append(current_slots)
		
		utterance_images = []		
		if 'images' in utterance and utterance['images'] not in [[], None, ""]:
			utterance_images = [url for url in utterance['images'] if url is not None]
		
		#Make a training instance out of the dialogue till now, with last utterace as prediction
		if i > 0 and utterance_['speaker'] == "system" and is_prev_utterance_a_question == True and len(utterance_images) > 0:
			#Positive image
			pos_image = random.sample(utterance_images, 1)[0]

			correctness_list.append(correctness(pos_image, list_slots[-2]))

		if utterance_['type'] == 'question':	
			is_prev_utterance_a_question = True



# print(list_slots)

feature_total = [0 for feature in features]
feature_correct = [0 for feature in features]
feature_acc = [0 for feature in features]

for instance in correctness_list:
	for f, feature in enumerate(instance):
		if feature == 1:
			feature_correct[f] += 1
		feature_total[f] += 1

for f, feature in enumerate(features):
	feature_acc[f] = feature_correct[f]/float(feature_total[f])

print(feature_acc)
print(sum(feature_correct)/float(len(features)))