import tensorflow as tf
from parameters import *
import pickle
from tensorflow.python.layers import core as layers_core

#Get embeddings of vocab
embeddings_google = pickle.load(open(data_dump_dir + 'vocab_embeddings.pkl', 'rb'))


class HREI():

	def __init__(self, vocab_size):
		self.vocab_size = vocab_size

		#Convert vocab embeddings into a tensor
		self.embedding_matrix = tf.constant(embeddings_google, name= "embedding_matrix")
		self.embedder = tf.get_variable("embedding_matrix", [self.vocab_size, word_embedding_size])
		


	def create_placeholder(self):
		#encoder_image_inputs is a max_context_len sized list of [batch_size * num_images_in_context * image_rep_size] tensors
		self.encoder_image_inputs = [tf.placeholder(tf.float32,[None, num_images_in_context, image_size], name="encoder_image_inputs") for i in range(max_context_len)]  
		#
		#encoder_text_inputs is a max_context_len sized list of [batch_size, max_utter_len] tensors
		self.encoder_text_inputs = [tf.placeholder(tf.int32,[None, max_utter_len], name="encoder_text_inputs") for j in range(max_context_len)] 
		#
		#A is  [num_nodes, num_nodes]
		self.A = tf.placeholder(tf.float32, [batch_size, num_nodes, num_nodes], name= "A")
		#
		#pos_images is [batch_size, 1, image_size] tensor
		self.pos_images = tf.placeholder(tf.float32, [None, 1, image_size], name="pos_images")
		#
		#negs_images is [batch_size, num_neg_images_use, image_size] tensor
		self.negs_images = tf.placeholder(tf.float32, [None, num_neg_images_use, image_size], name="negs_images")


		#
		#W1 is [cell_state_size, gcn_layer1_out_size]
		self.W1 = tf.Variable(tf.random_normal([cell_state_size, gcn_layer1_out_size], stddev=0.01), name = "W1")
		#
		#b1 is [gcn_layer1_out_size]
		self.b1 = tf.Variable(tf.random_normal([gcn_layer1_out_size], stddev=0.01), name = "b1")


		
	def sentence_encoder(self, encoder_text_input):
		# Look up embedding:
		# encoder_text_input: [batch_size, max_time]
		encoder_embedded_input = tf.nn.embedding_lookup(self.embedder, encoder_text_input)
		# encoder_embedded_input: [batch_size, max_time, embedding_size]
			
		with tf.variable_scope('enc1', reuse = tf.AUTO_REUSE):
			#RNN cell converts [batch_size, max_utter_len, word_embedding_size] to [batch_size, cell_state_size]
			self.cell_enc1_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
			self.cell_enc1_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)

			# Run Dynamic RNN
			(sentence_output_1, sentence_output_2), (sentence_state_1, sentence_state_2) = \
			tf.nn.bidirectional_dynamic_rnn(self.cell_enc1_fw, self.cell_enc1_bw, encoder_embedded_input,
		 									time_major=False, dtype=tf.float32)

		sentence_output_stack = tf.stack([sentence_output_1, sentence_output_2])
		sentence_output_mean = tf.reduce_mean(sentence_output_stack,0)
		sentence_state_stack = tf.stack([sentence_state_1, sentence_state_2])
		sentence_state_mean = tf.reduce_mean(sentence_state_stack,0)
		# sentence_output_mean: [batch_size, max_utter_len, cell_state_size]
		# sentence_state_mean: [batch_size, cell_state_size]		
		return sentence_output_mean, sentence_state_mean




	
	def image_encoder(self, encoder_image_inputs):
		#encoder_image_inputs - [batch_size, image_size]
		with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):
			encoder_image_states = tf.contrib.layers.fully_connected(encoder_image_inputs, 
												image_embedding_size, 
												activation_fn = None)
		#encoder_image_states - [batch_size, image_embedding_size]		
		return encoder_image_states
		

		

	#Create an adjacency matrix with encoded text and images
	def gcn(self, encoder_text_states, encoder_image_states): 

		# Use encoder_text_states and encoder_image_states to make a graph of nodes (X) and edges (A)
		# encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		# encoder_image_states is max_context_len * [batch_size, num_images_in_context, cell_state_size] tensors
		text = tf.stack(encoder_text_states, axis = 1)
		images = tf.concat(encoder_image_states, axis = 1)
		# text is [batch_size, max_dialogue_len, cell_state_size]
		# images is [batch_size, max_dialogue_len * num_images_in_context, cell_state_size]
		X = tf.concat([text, images], axis = 1)
		# X is [batch_size, num_nodes, cell_state_size]
		
		# A is the adjacency matrix - [batch_size, num_nodes, num_nodes]
		# X is the node embedding matrix - [batch_size, num_nodes, cell_state_size]

		W1_batch = tf.stack([self.W1]*batch_size, axis = 0)
		b1_batch = tf.stack([tf.stack([self.b1]*num_nodes, axis = 0)]*batch_size, axis = 0)
		gcn_layer1 = tf.nn.relu(tf.add(tf.matmul(tf.matmul(self.A, X), W1_batch), b1_batch))
		#gcn_layer1 is [num_nodes, gcn_layer1_out_size]

		return gcn_layer1





	def add_and_connect(self, nodes):
		sum_nodes = tf.reduce_sum(nodes, axis = 1)
		layer1 = tf.contrib.layers.fully_connected(sum_nodes, 
												image_embedding_size)
		
		return layer1

	





	#Encode each utterance. Append all utterance states. 
	def hierarchical_encoder(self):		
		#encoder_text_inputs is max_context_len * [batch_size, max_utter_len] tensors
		#encoder_image_inputs is max_context_len * [batch_size, image_size] tensors
		
		encoder_text_states = []
		for encoder_text_input in self.encoder_text_inputs:
			encoder_text_output, encoder_text_state = self.sentence_encoder(encoder_text_input)
			encoder_text_states.append(encoder_text_state)
		# encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		
		encoder_image_states = []
		for encoder_image_input in self.encoder_image_inputs:			
			encoder_image_state = self.image_encoder(encoder_image_input)
			encoder_image_states.append(encoder_image_state)
		# encoder_image_states is max_context_len * [batch_size, num_images_in_context, cell_state_size] tensors
		
		context_state = self.gcn(encoder_text_states, encoder_image_states)
		# context_state is [batch_size, cell_state_size] tensor	
		final_state = self.add_and_connect(context_state)
		# final_state is [batch_size, cell_state_size] tensor	

		return final_state



	def cosine_similarity(self, image_set_1, image_set_2): 
	   	with tf.variable_scope('cossim', reuse = tf.AUTO_REUSE):
	   		#image_set_1 is of dimension (batch_size * image_embedding_size)
	   		#image_set_2 is of dimension (batch_size * image_embedding_size)
		  	normed_1 = tf.nn.l2_normalize(image_set_1, 1)
		   	normed_2 = tf.nn.l2_normalize(image_set_2, 1) 
		   	#normed_1 is of dimension batch_size * image_embedding_size
		   	#normed_2 is of dimension batch_size * image_embedding_size
		   	cosine_sim = tf.matmul(normed_1, normed_2, transpose_b=True)
		   	cosine_sim = tf.diag_part(cosine_sim)
		   	#cosine_similarity is of dimension batch_size * batch_size
	   	return cosine_sim

	

	def loss(self, context_state):
		#context_state is [batch_size, image_embedding_size]
		
		#pos_images is [batch_size, 1, image_size]
		pos_images = tf.reshape(self.pos_images, [batch_size, image_size])
		#pos_images is now [batch_size, image_size]
		encoded_pos_images = self.image_encoder(pos_images)
		#encoded_pos_images is [batch_size, cell_state_size]
		cosine_sim_pos = self.cosine_similarity(context_state, encoded_pos_images)
		#cosine_sim_pos is of [batch_size]

		#negs_images is [batch_size, num_neg_images, image_size]
		negs_images_reshaped = tf.transpose(self.negs_images, [1, 0, 2])
		#negs_image_reshaped is [num_neg_images, batch_size, image_size]
		# encoded_negs_images = []
		# #encoded_negs_images will be num_neg_images * [batch_size, cell_state_size]
		# cosine_sim_negs = []
		# #cosine_sim_negs will be num_neg_images * [batch_size]

		losses = []
		ones = tf.ones([batch_size])
		zeros = tf.zeros([batch_size])
		for neg_images_reshaped in tf.unstack(negs_images_reshaped):
			encoded_neg_images = self.image_encoder(neg_images_reshaped)
			#encoded_neg_images is [batch_size, cell_state_size]
			cosine_sim_neg = self.cosine_similarity(context_state, encoded_neg_images)
			#cosine_sim_neg is [batch_size]
			# encoded_negs_images.append(encoded_neg_images)
			# cosine_sim_negs.append(cosine_sim_neg)	
			act_loss = ones - cosine_sim_pos + cosine_sim_neg
			per_loss = tf.maximum(zeros, act_loss)
			losses.append(per_loss)	
		loss = tf.reduce_sum(tf.add_n(losses))
  		return loss



	def train(self, losses):
		parameters = tf.trainable_variables()    
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08)
		global_step = tf.Variable(0,name="global_step",trainable='False')

		#Without gradient clipping
		#train_op=optimizer.minimize(losses,global_step=global_step)
		#With gradient clipping
		gradients = tf.gradients(losses,parameters)		
		clipped_gradients,norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		train_op = optimizer.apply_gradients(zip(clipped_gradients,parameters), global_step = global_step)

		return train_op, clipped_gradients




	def sim(self, context_state):
		#context_state is [batch_size, image_embedding_size]

		#pos_images is [batch_size, image_size]
		encoded_pos_images = self.image_encoder(self.pos_images)
		#encoded_pos_images is [batch_size, image_embedding_size]
		pos_image_reshaped = tf.reshape(encoded_pos_images, [batch_size, cell_state_size])
		#pos_image_reshaped is [batch_size, cell_state_size]		
		cosine_sim_pos = self.cosine_similarity(context_state, pos_image_reshaped)
		#cosine_sim_pos is of [batch_size]

		#negs_images is [batch_size, num_neg_images, image_size]
		negs_image_reshaped = tf.transpose(self.negs_images, [1, 0, 2])
		#negs_image_reshaped is [num_neg_images, batch_size, image_size]
		cosine_sim_negs = []
		#cosine_sim_negs will be num_neg_images * [batch_size]

		ones = tf.ones([batch_size])
		zeros = tf.zeros([batch_size])
		for image in tf.unstack(negs_image_reshaped):			
			encoded_image = self.image_encoder(image)
			#encoded_image is [batch_size, cell_state_size]
			cosine_sim_neg = self.cosine_similarity(context_state, encoded_image)
			#cosine_sim_neg is [batch_size]
			cosine_sim_negs.append(cosine_sim_neg)

		#cosine_sim_negs is list of num_neg_images * [batch_size]
		cosine_sim_neg_tensor = tf.stack(cosine_sim_negs)
		#cosine_sim_neg_tensor is [num_neg_images, batch_size]
		cosine_sim_neg_tensor_transposed = tf.transpose(cosine_sim_neg_tensor, [1,0])
		#cosine_sim_neg_tensor_transposed is [batch_size, num_neg_images]

   		return cosine_sim_pos, cosine_sim_neg_tensor_transposed