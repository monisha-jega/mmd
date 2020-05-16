import pickle
from parameters_domain_model import *


image_kg = pickle.load(open(data_dump_dir + "image_kg.pkl"))

data = []
count = 0
for image_url, image_features in image_kg.items():
	gender = image_features[3]
	if gender == "men":
		v = 0
	elif gender == "women":
		v = 1
	elif gender == "kids":
		v = 2
	elif gender == "all":
		v = 3
	else:
		v = 3
	data.append([image_url, v])

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



