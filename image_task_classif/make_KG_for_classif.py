from __future__ import print_function
import json, os, pickle
from parameters_classif import *


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
					#['type'],
					#['fit'],
					#['brand'],
					['gender'],
					#['neck'],
					['material', 'fabric'],
					#['length'],
					#['sleeves'],
					#['model_worn'],
					#['currency'],
					['color']
				 ]
num_features = len(domain_features)
feature_existence_count = [0 for i in range(num_features)]


dirlist = [public_jsons_dir + name for name in ['public_jsons', 'public_jsons (2)', 'public_jsons (3)', 'public_jsons (4)']]
count = 0
KG = {}
puncs = ['\\', '/', '|', '(', '-', ',']	

for diri in dirlist[:1]:
	print(len(os.listdir(diri)))
	for json_file in os.listdir(diri)[:]:
		#print("ok")
		the_json = convert_json(json.load(open(diri +"/" + json_file)))
		feature_vec = ["" for i in range(num_features)]

		for l in range(num_features):
			feature_names = domain_features[l]
			for feature_name in feature_names:
				if feature_name in the_json:
					if the_json[feature_name] == "":
					# or (l == 0 and (not is_int(the_json[feature_name]) or int(the_json[feature_name]) == 0)):
						pass
					else:
						try:
							feature_text = the_json[feature_name].strip().split()[0].lower()
							#print(feature_text)
							for punc in puncs:
								feature_text = feature_text.split(punc)[0]

							feature_vec[l] = feature_text
							feature_existence_count[l] += 1
						except:							
							pass

		KG[the_json["image_filename"]] = feature_vec
		for orientation, links in the_json['image_filename_all'].items():
			for link in links:
				KG[link] = feature_vec
		

		count += 1
		if count%20000 == 0:
			print(count, end="")
			print(" ")
	print()
		
color_count = 0
color_index_map = {}
color_count_map = {}
mat_index_map = {}	
mat_count_map = {}	
mat_count = 0		
for url, vec in KG.items():
	#color is vec[2], mat is vec[1]
	color_name, mat_name = vec[2], vec[1]
	if color_name not in color_count_map:
		color_count_map[color_name] = 1
	else:
		color_count_map[color_name] += 1
	if mat_name not in mat_count_map:
		mat_count_map[mat_name] = 1
	else:
		mat_count_map[mat_name] += 1	
print(len(color_count_map.items()), len(mat_count_map.items()))

#Delete misc colors
for color, count in color_count_map.items():
	if count < 1000:
		del color_count_map[color]

#Delete misc mats
for mat, count in mat_count_map.items():
	if count < 1000:
		del mat_count_map[mat]

#Replace misc colors and mats with "misc"
for url, vec in KG.items():
	#color is vec[1], mat is vec[2]
	color_name, mat_name = vec[2], vec[1]
	if color_name not in color_count_map:
		vec[2] = "misc"
	if mat_name not in mat_count_map:
		vec[1] = "misc"

print("Number of gender classes : ", feature_existence_count[0])
print("Number of color classes : ", len(color_count_map.items()))
print("Number of mat classes : ", len(mat_count_map.items()))


pickle.dump(KG, open(data_dump_dir+"image_kg_classif.pkl", "wb"))
		
