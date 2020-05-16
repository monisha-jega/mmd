from __future__ import print_function
import json, os, pickle
from parameters import *

#Convert keys to lowercase
def lowercase_json(the_json):
	new_json = {}
	for key, val in the_json.items():		
		if key.lower() == "fabric": 
			key = "material"
		new_json[key.lower()] = val
	return new_json

#Check if integer
def is_integer(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

link1 = "https://images-na.ssl-images-amazon.com/images/I/4117JORoryL.jpg"
link2 = "https://images-na.ssl-images-amazon.com/images/I/512ZS1-uQcL.jpg"

################################################## PARAMETERS ##################################################

#All image features
domain_features = [
					'price', 					
					'currency',
					'style',  
					'type',
					'fit',
					'neck',
					'length',
					'sleeves',
					'color',
					'material',
					'brand',
					'gender'
				 ][2:]

KG = {}
value_existence_count = {}
unique_value_count = {}
freq_value_count = {}
feature_index_map = {}

puncs = ['\\', '/', '|', '(', '-', ',']

# root_dir = "../../raw_catalog/"
# dirlist = [root_dir + name for name in ['public_jsons', 'public_jsons (2)', 'public_jsons (3)', 'public_jsons (4)']]
dirlist = [public_jsons]

for feature_key in domain_features:
	value_existence_count[feature_key] = 0
	unique_value_count[feature_key] = 1
	feature_index_map[feature_key] = {unk_feature_val : unk_feature_id}
	freq_value_count[feature_key] = {}

count = 0
for diri in dirlist[:]:
	print("Number of image json files : ", len(os.listdir(diri)))
	for json_file in os.listdir(diri)[:]:
		
		the_json = lowercase_json(json.load(open(diri +"/" + json_file)))
		feature_dic = {}
		for feature_key in domain_features:			
			
			if feature_key in the_json and the_json[feature_key] not in [None, ""]:
				try: 
					feature_val = the_json[feature_key]
					#If integer value
					# if is_integer(feature_val):
					# 	feature_dic[feature_key] = feature_val						
					# 	value_existence_count[feature_key] += 1
					# else:
					#Process text
					if feature_val not in [None, ""]:
						feature_text = feature_val.strip().lower().split()[0]
						for punc in puncs:
							feature_text = feature_text.split(punc)[0]
					
					feature_dic[feature_key] = feature_text
					value_existence_count[feature_key] += 1

					#Update feature index and number of uniqe values
					if feature_text not in feature_index_map[feature_key]:
						freq_value_count[feature_key][feature_text] = 1
						feature_index_map[feature_key][feature_text] = unique_value_count[feature_key]
						unique_value_count[feature_key] += 1
					else:
						freq_value_count[feature_key][feature_text] += 1

				except Exception as e:
					feature_dic[feature_key] = unk_feature_val
					print("Caugh exception", e)	
			else:
				feature_dic[feature_key] = unk_feature_val
						

		#Set same attributes to all orientations
		image_links = [the_json["image_filename"]]
		for orientation, links in the_json['image_filename_all'].items():
			image_links += links 
		for link in image_links:
			if link == link1:
				print("link1 o=dfd")
			if link == link2:
				print("link2 o=dfd")
			KG[link] = feature_dic		

		count += 1
		if count%20000 == 0:
			print(count)
			
print(value_existence_count)
print(unique_value_count)

pickle.dump(unique_value_count, open(common_data_dump_dir+"unique_value_count.pkl", "wb"))
pickle.dump(value_existence_count, open(common_data_dump_dir+"value_existence_count.pkl", "wb"))
pickle.dump((KG, feature_index_map, freq_value_count), open(common_data_dump_dir + "temp", 'wb'))
print("dumped 1st round")


(KG, feature_index_map, freq_value_count) = pickle.load(open(common_data_dump_dir + "temp", 'rb'))
print(len(KG.items()))


for feature_num, feature_key in enumerate(domain_features):
	new_index = 1 #(as 1 is reserved for unk)
	for feature_val, feature_count in freq_value_count[feature_key].items():
		if feature_count < feature_min_freq_thresholds[feature_num]:
			del feature_index_map[feature_key][feature_val]
		else:			
			feature_index_map[feature_key][feature_val] = new_index 
			new_index += 1
	print(new_index-1)
print("removed rare features")


#Remap indices
for url, feature_dic in KG.items():
	new_feature_dic = {}
	for feature_key in domain_features:	
		feature_val = feature_dic[feature_key]
		try:
			new_feature_dic[feature_key] = feature_index_map[feature_key].get(feature_val, unk_feature_id)
		except Exception as e:
			print(e, feature_key, feature_val, feature_dic)
	KG[url] = new_feature_dic
print("remapped")


print(link1 in KG)
print(link2 in KG)

pickle.dump(KG, open(kg_file, "wb"))
pickle.dump(feature_index_map, open(feature_index_map_file, "wb"))
pickle.dump(freq_value_count, open(common_data_dump_dir+"freq_value_count.pkl", "wb"))
print(len(KG.items()))
