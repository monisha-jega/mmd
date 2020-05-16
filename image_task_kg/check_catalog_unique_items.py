from __future__ import print_function
import json, os, pickle



def convert_json(the_json):
	new_json = {}
	for key, val in the_json.items():
		new_json[key.lower()] = val
	return new_json
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False



domain_features = [
					#['price'],
					#['style'],
					['type'],
					['fit'],
					['brand'],
					['gender'],
					#['neck'],
					['material', 'fabric'],
					#['length'],
					#['sleeves'],
					#['model_worn'],
					#['currency'],
					['color']
				 ]

puncs = ['\\', '/', '|', '(', '-', ',']	

num_features = len(domain_features)
feature_existence_count = [0 for i in range(num_features)]
feature_vals = [{} for i in range(num_features)]

root_dir = "../../raw_catalog/"
dirlist = [root_dir + name for name in ['public_jsons', 'public_jsons (2)', 'public_jsons (3)', 'public_jsons (4)']]
count = 0

for diri in dirlist[:]:
	print(len(os.listdir(diri)))
	for json_file in os.listdir(diri):
		#print("ok")
		the_json = convert_json(json.load(open(diri +"/" + json_file)))

		for l in range(num_features):
			feature_names = domain_features[l]
			for feature_name in feature_names:
				if feature_name in the_json:
					if the_json[feature_name] == "":
					#or (l == 0 and (not is_int(the_json[feature_name]) or int(the_json[feature_name]) == 0)):
						#print("Passs")
						pass
					else:
						feature_existence_count[l] += 1
						#print(feature_text)
						try:
							feature_text = the_json[feature_name].strip().split()[0].lower()
							for punc in puncs:
								feature_text = feature_text.split(punc)[0]

							if feature_text in feature_vals[l]:
								feature_vals[l][feature_text] += 1
							else:
								#print(feature_text)
								feature_vals[l][feature_text] = 1
						except Exception as e:
							print(e)
							pass

		

		count += 1
		if count%20000 == 0:
			print(count, end="")
	print()
			
print(feature_existence_count)
for dic in feature_vals:
	#print(dic)
	print("Unique values", len(dic.items()))
	images = 0
	for k, v in dic.items():
		images += v
	print("Total images", images)


json.dump(feature_vals, open("../../features.json", 'w'))
pickle.dump(feature_vals, open("../../features.pkl", 'wb'))
#print(feature_vals)
		