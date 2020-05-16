import json, os

dialogue_states = {}
count = 0


direc = '../../dataset/v1/train/'
print(len(os.listdir(direc)))
for filename in os.listdir(direc):
	the_json = json.load(open(direc + filename))
	for utterance in the_json:
		the_type = utterance["type"]
		if the_type in dialogue_states:
			dialogue_states[the_type] += 1
		else:
			dialogue_states[the_type] = 0

	count += 1
	if count % 10000 == 0:
		print(count)
		print(dialogue_states)

