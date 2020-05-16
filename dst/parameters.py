train_dir = "/scratch/scratch2/monishaj/dataset/v1/train/"
val_dir = "/scratch/scratch2/monishaj/dataset/v1/valid/"
test_dir= "/scratch/scratch2/monishaj/dataset/v1/test/"
data_dump_dir = "data_dump/"
#data_dump_dir = "/scratch/scratch2/monishaj/dst_data_dump/"
model_dump_dir = "/scratch/scratch2/monishaj/dst_model_dump/"
#Word embeddings file  
embed_file = '../../GoogleNews-vectors-negative300.bin'

start_word = "</s>"
start_word_id = 0
end_word = "</e>"
end_word_id = 1
pad_word = "<pad>"
pad_word_id = 2
unk_word = "<unk>"
unk_word_id = 3


MAX_SEQUENCE_LENGTH = 30
NUM_CLASSES = 80

vocab_freq_cutoff = 1

cell_state_size = 512
hidden_units = [100, 50]
word_embedding_size = 300
batch_size = 5
epochs = 10

restore_trained = False