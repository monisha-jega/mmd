from __future__ import print_function
import json, os, pickle
from parameters import *


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
					['price'],
					#['style'],
					['type'],
					['fit'],
					#['brand'],
					['gender'],
					#['neck'],
					['material', 'fabric'],
					#['length'],
					#['sleeves'],
					#['model_worn'],
					['currency'],
					['color']
				 ]
num_features = len(domain_features)
feature_existence_count = [0 for i in range(num_features)]


root_dir = "../../raw_catalog/"
dirlist = [root_dir + name for name in ['public_jsons', 'public_jsons (2)', 'public_jsons (3)', 'public_jsons (4)']]
count = 0
KG = {}

for diri in dirlist[:]:
	print(len(os.listdir(diri)))
	for json_file in os.listdir(diri):
		#print("ok")
		the_json = convert_json(json.load(open(diri +"/" + json_file)))
		feature_vec = ["" for i in range(num_features)]

		for l in range(num_features):
			feature_names = domain_features[l]
			for feature_name in feature_names:
				if feature_name in the_json:
					if the_json[feature_name] == "" or (l == 0 and (not is_int(the_json[feature_name]) or int(the_json[feature_name]) == 0)):
						pass
					else:
						feature_vec[l] = the_json[feature_name]
						feature_existence_count[l] += 1

		KG[the_json["image_filename"]] = feature_vec
		for orientation, links in the_json['image_filename_all'].items():
			for link in links:
				KG[link] = feature_vec
		

		count += 1
		if count%20000 == 0:
			print(count, end="")
			print(" ")
	print()
			
print(feature_existence_count)


json.dump(KG, open(data_dump_dir+"image_kg.json", "wb"))
pickle.dump(KG, open(data_dump_dir+"image_kg.pkl", "wb"))
		