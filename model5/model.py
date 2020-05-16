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

#encoder_image_inputs is a max_context_len sized list of [batch_size * image_rep_size] tensors
image_inputs = [tf.placeholder(tf.float32,[None, image_size], name="image_inputs") for i in range(num_images_per_utterance)]  
#pos_images is [batch_size, 1, image_size] tensor
pos_images = tf.placeholder(tf.float32, [None, 1, image_size], name="pos_images")
#negs_images is [batch_size, num_neg_images, image_size] tensor
negs_images = tf.placeholder(tf.float32, [None, num_neg_images_use, image_size], name="negs_images")



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
		(sentence_outputs_1, sentence_outputs_2), (sentence_state_1, sentence_state_2) = \
		tf.nn.bidirectional_dynamic_rnn(cell_enc1_fw, cell_enc1_bw, encoder_embedded_sentence,
	 									time_major=False, dtype=tf.float32)

		sentence_state_stack = tf.stack([sentence_state_1, sentence_state_2])
		sentence_state_mean = tf.reduce_mean(sentence_state_stack,0)
		# sentence_outputs_stack = tf.stack([sentence_output_1, sentence_output_2])
		# sentence_outputs_mean = tf.reduce_mean(sentence_outputs_stack,0)
		return sentence_state_mean

def image_encoder(image_vgg):
	#Input - [batch_size, image_size]
	#Output - [batch_size, image_embedding_size]
	with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):
		encoded_image = tf.contrib.layers.fully_connected(image_vgg,
													image_embedding_size, 
													activation_fn = None)
	return encoded_image



#CONCAT ALL
def hierarchical_and_concat():

	#Level 1 encoder
	sentence_encodings = []
	for text_input in text_inputs_ph:
	    sentence_encoding = sentence_encoder(text_input)
	    sentence_encodings.append(sentence_encoding)
	#sentence_encodings is max_context_len * [batch_size, cell_state_size]
	# sentence_encodings = tf.stack(sentence_encodings, axis=1)
	# #sentence_encodings is now [batch_size, max_context_len, cell_state_size]
	sentence_encodings_concat = tf.concat(sentence_encodings, axis=1)
	#sentence_encodings_concat is now [batch_size, max_context_len * cell_state_size]

	encoded_image_states = []
	for image in image_inputs:			
		encoded_image_state = image_encoder(image)
		encoded_image_states.append(encoded_image_state)
	#encoded_image_states is num_images_per_utterance * [batch_size, cell_state_size] tensors	

	#encoded_image_states_concat is now [batch_size, num_images_per_utterance * cell_state_size]
	encoded_image_states_concat = tf.concat(encoded_image_states, axis = 1)

	final_features = tf.concat([sentence_encodings_concat, encoded_image_states_concat, last_ds_ph], axis = 1)
	return final_features



#MLP 
def MLP(final_features):
	layer1 = tf.layers.dense(final_features, hidden_units[0], name = "layer1")
	output_layer = tf.layers.dense(layer1, image_embedding_size, name = "output_layer")
	return output_layer
	


def cosine_similarity(image_set_1, image_set_2):
   	#image_set_1 is of dimension (batch_size * image_embedding_size)
   	#image_set_2 is of dimension (batch_size * image_embedding_size)

   	with tf.variable_scope('cossim', reuse = tf.AUTO_REUSE):

   		normed_1 = tf.nn.l2_normalize(image_set_1, 1)
	   	normed_2 = tf.nn.l2_normalize(image_set_2, 1) 
	   	#normed_1 is of dimension batch_size * image_embedding_size
	   	#normed_2 is of dimension batch_size * image_embedding_size

	   	cosine_sim = tf.matmul(normed_1, normed_2, transpose_b=True)
	   	cosine_sim = tf.diag_part(cosine_sim)
	   	#cosine_similarity is of dimension batch_size * batch_size
   	return cosine_sim

	

def loss_train(context_state):
	#context_state is [batch_size, image_embedding_size]
	#pos_images is [batch_size, 1, image_size]

	# with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):

	pos_images_ = tf.reshape(pos_images, [batch_size, image_size])
	#pos_images is now [batch_size, image_size]
	encoded_pos_images = image_encoder(pos_images_)
	#encoded_pos_images is [batch_size, image_embedding_size]
	#image_embedding_size	
	cosine_sim_pos = cosine_similarity(context_state, encoded_pos_images)
	#cosine_sim_pos is of [batch_size]

	negs_images_reshaped = tf.transpose(negs_images, [1, 0, 2])
	#negs_image_reshaped is [num_neg_images, batch_size, image_size]
	encoded_negs_images = []
	#encoded_negs_images is [num_neg_images, batch_size, image_embedding_size]
	cosine_sim_negs = []
	#cosine_sim_negs is [num_neg_images, batch_size]

	losses = []
	act_losses = []
	ones = tf.ones([batch_size])
	zeros = tf.zeros([batch_size])

	for neg_images_reshaped in tf.unstack(negs_images_reshaped):

		encoded_neg_images = image_encoder(neg_images_reshaped)
		#encoded_neg_images is [batch_size,image_embedding_size]
		#image_embedding_size

		cosine_sim_neg = cosine_similarity(context_state, encoded_neg_images)
		encoded_negs_images.append(encoded_neg_images)
		cosine_sim_negs.append(cosine_sim_neg)
		#cosine_sim_neg is of [batch_size]

		act_loss = ones - cosine_sim_pos + cosine_sim_neg		
		act_losses.append(act_loss)	
		per_loss = tf.maximum(zeros, act_loss)
		losses.append(per_loss)

	total_loss = tf.reduce_sum(tf.add_n(losses))
		
	parameters=tf.trainable_variables()    
	optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
	global_step=tf.Variable(0,name="global_step",trainable='False')

	#No gradient clipping
	#train_op=optimizer.minimize(losses,global_step=global_step)

	#With gradient clipping
	gradients=tf.gradients(losses,parameters)		
	clipped_gradients,norm=tf.clip_by_global_norm(gradients, max_gradient_norm)
	train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)

	return total_loss, train_op




def sim_func(context_state):
	#context_state is [batch_size, image_embedding_size]
	#pos_images is [batch_size, image_size]
	
	#ith tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):
		
	pos_images_ = tf.reshape(pos_images, [batch_size, image_size])
	#pos_images is now [batch_size, image_size]
	encoded_pos_images = image_encoder(pos_images_)

	#pos_image_reshaped = tf.reshape(encoded_pos_images, [batch_size, image_embedding_size])
	cosine_sim_pos = cosine_similarity(context_state, encoded_pos_images)
	#pos_image_reshaped is [batch_size, image_embedding_size]
	#cosine_sim_pos is of [batch_size]

	negs_image_reshaped = tf.transpose(negs_images, [1, 0, 2])
	#negs_image_reshaped is [num_neg_images, batch_size, image_size]
	
	cosine_sim_negs = []
	for image in tf.unstack(negs_image_reshaped):
		
		encoded_image = image_encoder(image)
		#encoded_image is [batch_size, image_embedding_size]
		cosine_sim_neg = cosine_similarity(context_state, encoded_image)
		#cosine_sim_neg is of [batch_size]
		cosine_sim_negs.append(cosine_sim_neg)

	#cosine_sim_neg is list of num_neg_images * [batch_size]
	cosine_sim_neg_tensor = tf.stack(cosine_sim_negs)
	#cosine_sim_neg_tensor is [num_neg_images, batch_size]
	cosine_sim_neg_tensor_transposed = tf.transpose(cosine_sim_neg_tensor, [1,0])
	#cosine_sim_neg_tensor_transposed is [batch_size, num_neg_images]

	return cosine_sim_pos, cosine_sim_neg_tensor_transposed




















#SAVER
def saver():
	return tf.train.Saver()

		