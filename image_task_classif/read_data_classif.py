import pickle, json
from parameters_classif import *


image_kg = pickle.load(open(data_dump_dir + "image_kg_classif.pkl"))
color_count, mat_count = 1, 1
color_map = {"misc" : 0, "" : 0}
mat_map = {"misc" : 0, "" : 0}


data = []
count = 1
for image_url, image_features in image_kg.items():

	gender = image_features[0]
	v = 3 #for all and misc
	if gender == "men":
		v = 0
	elif gender == "women":
		v = 1
	elif gender == "kids":
		v = 2

	color_name = image_features[1]
	if color_name in color_map:
		color = color_map[color_name]	
	else:
		color_map[color_name] = color_count		
		color = color_count
		color_count += 1

	mat_name = image_features[2]
	if mat_name in mat_map:
		mat = mat_map[mat_name]	
	else:
		mat_map[mat_name] = mat_count		
		mat = mat_count
		mat_count += 1

	data.append([image_url, v, color, mat])

	count += 1
	if count % 10000 == 0:
		print(count)


json.dump(color_map, open(data_dump_dir + "color_map.json", 'w'))
print("Number of colors : ", color_count)
json.dump(mat_map, open(data_dump_dir + "mat_map.json", 'w'))
print("Number of mats : ", mat_count)

data_size = len(data)
train_size = int(data_size * 0.7)
val_test_size = int(data_size * 0.15)

train_data = data[:train_size]
val_data = data[train_size : train_size + val_test_size]
test_data = data[train_size + val_test_size:]


pickle.dump(train_data, open(data_dump_dir + "train_kg.pkl", 'wb'))
pickle.dump(val_data, open(data_dump_dir + "val_kg.pkl", 'wb'))
pickle.dump(test_data, open(data_dump_dir + "test_kg.pkl", 'wb'))



