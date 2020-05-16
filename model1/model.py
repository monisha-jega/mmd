import tensorflow as tf
import pickle
from parameters import *



#PLACEHOLDERS
#Text inputs
text_inputs_ph = [tf.placeholder(tf.int32,[None, max_utter_len], name="text_inputs") for j in range(max_context_len)] 
#Dialogue final slots
dialogue_slots_ph = [tf.placeholder(tf.float32,[None, feature_size], name="image_reps") for feature_size in feature_sizes]
#State of last dialogue
last_ds_ph =tf.placeholder(tf.float32,[None, num_ds], name="last_ds")
#Final image representations for num_images_per_utterance images
image_reps_ph = [[tf.placeholder(tf.float32,[None, feature_size], name="image_reps") for feature_size in feature_sizes] for i in range(num_images_per_utterance)] 

#Output image representation
target_image_rep_ph = [tf.placeholder(tf.int32,[None], name="target_image_rep") for feature_size in feature_sizes]


def embedding_mat():
	#LOAD EMBEDDINGS	
	google_embeddings_for_vocab = pickle.load(open(vocab_embeddings_file, 'rb'))
	vocab_size = len(google_embeddings_for_vocab)	
	google_embeddings_for_vocab_mat = tf.constant(google_embeddings_for_vocab, name= "google_embeddings_for_vocab")
	embedding_tensor = tf.get_variable("embedding_tensor", [vocab_size, embedding_size])
	return embedding_tensor



#HRE
def sentence_encoder(text_input_sentence):	
	#Input - [batch_size, max_utter_len]
	#Output - [batch_size, cell_state_size]
	with tf.variable_scope('enc1', reuse = tf.AUTO_REUSE):
		#Convert [batch_size, max_utter_len] into [batch_size, max_utter_len, word_embedding_size]
		#Look up embedding:
		embedding_tensor = embedding_mat()
		encoder_embedded_sentence = tf.nn.embedding_lookup(embedding_tensor, text_input_sentence)	
		cell_enc1_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
		cell_enc1_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)
		#RNN cell converts [batch_size, max_utter_len, word_embedding_size] to [batch_size, cell_state_size]
		# sentence_output: [batch_size, max_utter_len, cell_state_size]
		# sentence_state: [batch_size, cell_state_size]		
		(sentence_output_1, sentence_output_2), (sentence_state_1, sentence_state_2) = \
		tf.nn.bidirectional_dynamic_rnn(cell_enc1_fw, cell_enc1_bw, encoder_embedded_sentence,
	 									time_major=False, dtype=tf.float32)
		sentence_state_stack = tf.stack([sentence_state_1, sentence_state_2])
		sentence_state_mean = tf.reduce_mean(sentence_state_stack,0)
		return sentence_state_mean

#Encode the context
def context_encoder(sentence_encodings):
	#encoder_context__input is max_context_len * [batch_size, cell_state_size + image_embedding_size] tensors	
	with tf.variable_scope('enc2'):
		cell_enc2_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
		cell_enc2_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)
		#RNN cell converts [batch_size, max_context_len, cell_state_size + image_embedding_size] to [batch_size, max_context_len, cell_state_size] and returns
		# context_output: [batch_size, max_context_len, cell_state_size]
		# context_state: [batch_size, cell_state_size]
		(context_output_1, context_output_2), (context_state_1, context_state_2) = \
		tf.nn.bidirectional_dynamic_rnn(cell_enc2_fw, cell_enc2_bw, sentence_encodings,
	 												time_major=False, dtype=tf.float32)
	context_state_stack = tf.stack([context_state_1, context_state_2])
	context_state_mean = tf.reduce_mean(context_state_stack,0)
	return context_state_mean



#CONCAT ALL
def hierarchical_and_concat():

	#Level 1 encoder
	sentence_encodings = []
	for text_input in text_inputs_ph:
	    sentence_encoding = sentence_encoder(text_input)
	    sentence_encodings.append(sentence_encoding)
	#sentence_encodings is max_context_len * [batch_size, cell_state_size]
	sentence_encodings = tf.stack(sentence_encodings, axis=1)
	#sentence_encodings is now [batch_size, max_context_len, cell_state_size]

	#Level 2 encoder - hre_state is [batch_size, cell_state_size]
	hre_state = context_encoder(sentence_encodings) 

	#Concatenate
	dialogue_slots_concat = tf.concat(dialogue_slots_ph, axis = 1)
	image_reps_features_concat = []
	for each_image_rep in image_reps_ph:
		image_reps_features_concat .append(tf.concat(each_image_rep, axis = 1))
	all_image_reps = tf.concat(image_reps_features_concat, axis = 1)

	final_features = tf.concat([hre_state, all_image_reps, last_ds_ph], axis = 1)
	#final_features = tf.concat([last_ds_ph], axis = 1)
	return final_features



#MLP 
def MLP(final_features):
	layer1 = tf.layers.dense(final_features, hidden_units[0], name = "layer1")
	layer2 = tf.layers.dense(layer1, hidden_units[1], name = "layer2")
	output_features = []
	for f, feature_size in enumerate(feature_sizes):
		output_feature = tf.layers.dense(layer2, feature_size, name = "output_feature_" + features[f])
		output_features.append(output_feature)
	return output_features


#LOSS AND TRAIN_OP
def loss_func(output_features):
	total_loss = 0
	feature_losses = []
	for f, (target_feature, output_feature) in enumerate(zip(target_image_rep_ph, output_features)):
		feature_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_feature, logits = output_feature, name = "loss_" + features[f])
		total_loss += feature_loss
		feature_losses.append(feature_loss)

	optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate)
	train_op=optimizer.minimize(total_loss)

	return total_loss, train_op, feature_loss, target_feature, output_feature


#SAVER
def saver():
	return tf.train.Saver()

		