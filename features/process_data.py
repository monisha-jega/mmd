import pickle, json, os, sys
import numpy as np
from parameters import *

KG = pickle.load(open(kg_file))

data = []
count = 1
for url, feature_dic in KG.items():
	feature_wanted = feature_dic[feature]
	data.append((url, feature_wanted))
	count += 1
	if count % 10000 == 0:
		print(count)

train_size = int((1 - 2*val_test_split) * len(data))
val_test_size = int(val_test_split * len(data))

train_data = data[:train_size]
val_data = data[train_size : train_size+val_test_size]
test_data = data[train_size+val_test_size:]

print("data processed", len(train_data), len(val_data), len(test_data))

pickle.dump(train_data, open(data_dump_dir + "train.pkl", 'wb'))
pickle.dump(val_data, open(data_dump_dir + "val.pkl", 'wb'))
pickle.dump(test_data, open(data_dump_dir + "test.pkl", 'wb'))



