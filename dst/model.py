import tensorflow as tf
from parameters import *
import pickle


#Get vocab embeddings from pickle
embeddings_for_vocab = pickle.load(open(data_dump_dir + "vocab_embeddings_dst.pkl", 'rb'))
vocab_count = len(embeddings_for_vocab)







#Convert embeddings into tensor
embedding_matrix = tf.constant(embeddings_for_vocab, name= "embedding_matrix")
embedder = tf.get_variable("embedding_matrix", [vocab_count+1, word_embedding_size])

#text_inputs is [batch_size, MAX_SEQUENCE_LENGTH]
text_inputs = tf.placeholder(tf.int32,[None, MAX_SEQUENCE_LENGTH], name="text_inputs")
#target is [batch_size] tensor
targets = tf.placeholder(tf.int32,[None], name="targets")

		
# Convert [batch_size, MAX_SEQUENCE_LENGTH] into [batch_size, MAX_SEQUENCE_LENGTH, word_embedding_size]
# Look up embedding:
# encoder_text_input: [batch_size, MAX_SEQUENCE_LENGTH]
encoder_embedded_input = tf.nn.embedding_lookup(embedder, text_inputs)
# encoder_embedded_input: [batch_size, MAX_SEQUENCE_LENGTH, word_embedding_size]

with tf.variable_scope('enc', reuse = tf.AUTO_REUSE):
	#Bidirectional RNN cell converts [batch_size, MAX_SEQUENCE_LENGTH, word_embedding_size] to [batch_size, cell_state_size]
    cell_enc1_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
	cell_enc1_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)

    # Run Dynamic RNN
	(sentence_output_1, sentence_output_2), (sentence_state_1, sentence_state_2) = \
	tf.nn.bidirectional_dynamic_rnn(cell_enc1_fw, cell_enc1_bw, encoder_embedded_input,
 									time_major=False, dtype=tf.float32)

sentence_state_stack = tf.stack([sentence_state_1, sentence_state_2])
encoder_state = tf.reduce_mean(sentence_state_stack,0)
# sentence_output: [batch_size, MAX_SEQUENCE_LENGTH, cell_state_size]
# sentence_state: [batch_size, cell_state_size]

layer1 = tf.layers.dense(encoder_state, hidden_units[0], name="layer1")
layer2 = tf.layers.dense(layer1, hidden_units[1], name="layer2")
output_layer = tf.layers.dense(layer2, NUM_CLASSES, name="output_layer")

loss_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = output_layer)
optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op=optimizer.minimize(loss_node)

saver = tf.train.Saver()