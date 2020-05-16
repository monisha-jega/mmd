scratch_dir = "/scratch/scratch2/monishaj/"

common_data_dump_dir = "../common_data_dump/"
public_jsons = "../../raw_catalog/public_jsons"

#Must contain ImageUrlToIndex.pkl and annoy.ann
annoy_dir = '../../raw_catalog/image_annoy_index/'
kg_file = common_data_dump_dir + "KG.pkl"



unk_feature_val, unk_feature_id = "<unk_val>", 0
feature_sizes = [s+1 for s in [105, 187, 137, 101, 225, 32, 185, 233, 1734, 4]]
features = [ 'style',  
			'type',
			'fit',
			'neck',
			'length',
			'sleeves',
			'color',
			'material',
			'brand',
			'gender']
num_features = len(features)


#The feature to predict
feature = "gender"
feature_size = feature_sizes[features.index(feature)]

data_dump_dir = "data_dump_features_" + feature + "/"
model_dump_dir = "model_dump_features_" + feature + "/"
train_data_file = data_dump_dir + "train.pkl"
val_data_file = data_dump_dir + "val.pkl"
test_data_file = data_dump_dir+ "test.pkl"

hidden_units = [1000, 500, 1000]
val_test_split = 0.15
image_size = 4096
batch_size = 500		    			#best value - 64
learning_rate = 0.01    				#best value - 0.0004
epochs = 200							#best value - Early stopping

use_images = True					#If 0, will use 0s for image instead of loading from annoy file
restore_trained = False 			#If true, will restore latest model from checkpoint





