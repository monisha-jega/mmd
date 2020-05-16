from __future__ import print_function
import json, os



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
					['style'],
					['type'],
					['fit'],
					['brand'],
					['gender'],
					['neck'],
					['material', 'fabric'],
					['length'],
					['sleeves'],
					['model_worn'],
					['price'],
					['currency'],
					['color']
				 ]
num_features = len(domain_features)
feature_existence_count = [0 for i in range(num_features)]

wanted_url= "http://ecx.images-amazon.com/images/I/31WewpsjKdL.jpg"
root_dir = "../../raw_catalog/"
dirlist = [root_dir + name for name in ['public_jsons', 'public_jsons (2)', 'public_jsons (3)', 'public_jsons (4)']]
count = 0

for diri in dirlist[-1:]:
	print(len(os.listdir(diri)))
	for json_file in os.listdir(diri):
		#print("ok")
		the_json = convert_json(json.load(open(diri +"/" + json_file)))
		
		if the_json['image_filename'] == wanted_url:
			print("FOUND")
			break
		else:
			for orientation, links in the_json['image_filename_all'].items():
				if len(links) > 0 and links[0] == wanted_url:
					print("FOUND IN SUB")
					break

		# for l in range(num_features):
		# 	feature_names = domain_features[l]
		# 	for feature_name in feature_names:

		# 		if feature_name in the_json.keys():
		# 			if the_json[feature_name] == "" or (l == 0 and (not is_int(the_json[feature_name]) or int(the_json[feature_name]) == 0)):
		# 				pass
		# 			else:
		# 				feature_existence_count[l] += 1
				

		

		count += 1
		if count%20000 == 0:
			print(count, end="")
			print( )
	print()
			
print(feature_existence_count)
		