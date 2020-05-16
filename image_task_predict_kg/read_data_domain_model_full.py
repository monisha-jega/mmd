import pickle
from parameters_domain_model_full import *


image_kg = pickle.load(open(data_dump_dir + "image_kg_pred.pkl"))

data = []
count = 0
for image_url, image_features in image_kg.items()[:100]:
	gender = image_features[0]
	if gender == "men":
			g = 0
	elif gender == "women":
		g = 1
	elif gender == "kids":
		g = 2
	elif gender == "all":
		g = 3
	else:
		g = 3		
	data.append([image_url] + [g] + image_features[1:])

	count += 1
	if count % 10000 == 0:
		print(count)

data_size = len(data)
train_size = int(data_size * 0.7)
val_test_size = int(data_size * 0.15)

train_data = data[:train_size]
val_data = data[train_size : train_size + val_test_size]
test_data = data[train_size + val_test_size:]


pickle.dump(train_data, open(data_dump_dir + "train_kg.pkl", 'wb'))
pickle.dump(val_data, open(data_dump_dir + "val_kg.pkl", 'wb'))
pickle.dump(test_data, open(data_dump_dir + "test_kg.pkl", 'wb'))