scratch_dir = "/scratch/scratch2/monishaj/"

common_data_dump_dir = "../common_data_dump/"
data_dump_dir = "../model1/data_dump_sampled/"
model_dump_dir = "model_dump/"
train_dir = "../../dataset/v1/train/"
val_dir = "../../dataset/v1/valid/"
test_dir = "../../dataset/v1/test/"
public_jsons = "../../raw_catalog/public_jsons"
embed_file = "../../GoogleNews-vectors-negative300.bin"

annoy_dir = '../../raw_catalog/image_annoy_index/'

kg_file = common_data_dump_dir + "KG.pkl"
feature_index_map_file = common_data_dump_dir + "feature_index_map.pkl"
vocab_embeddings_file = data_dump_dir + "vocab_embeddings.pkl"
vocab_word_to_id_file = data_dump_dir + "vocab_word_to_id.pkl"
state_index_map_file = data_dump_dir + "state_index_map.pkl"
train_data_file = data_dump_dir + "train.pkl"
val_data_file = data_dump_dir + "val.pkl"
test_data_file = data_dump_dir+ "test.pkl"

sample_images = False
pad_symbol, pad_id = "<pad>", 0
unk_symbol, unk_id = "<unk>", 1
vocab_min_freq = 5
num_neg_images_sample = 100
num_neg_images_use = 5
max_context_len = 5
max_utter_len = 30
num_images_per_utterance = 5

unk_feature_val, unk_feature_id = "<unk_val>", 0
feature_min_freq_thresholds = [100 for i in range(10)][:]
feature_sizes = [s+1 for s in [105, 187, 137, 101, 225, 32, 185, 233, 1734, 4]][:] #1 is for unknown #Total 2750
features = [ 'style',  
			'type',
			'fit',
			'neck',
			'length',
			'sleeves',
			'color',
			'material',
			'brand',
			'gender'][:]
num_features = len(features)
num_ds = 80

image_size = 4096
embedding_size = 300
image_embedding_size = 512
hidden_units = [1000, 1000]
cell_state_size = 512
max_gradient_norm = 0.1
epochs = 3
learning_rate = 0.004
batch_size = 500

use_images = True
restore_trained = False

