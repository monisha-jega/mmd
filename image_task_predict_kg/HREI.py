import tensorflow as tf
from parameters import *
import pickle
from tensorflow.python.layers import core as layers_core

#Get embeddings of vocab
embeddings_google = pickle.load(open(data_dump_dir + 'vocab_embeddings.pkl', 'rb'))

class HREI():

	def __init__(self, vocab_size):
		self.vocab_size = vocab_size

		#Convert embeddings into tensor
		self.embedding_matrix = tf.constant(embeddings_google, name= "embedding_matrix")
		# Embedding
		self.embedder = tf.get_variable("embedding_matrix", [self.vocab_size, word_embedding_size])
		



	def create_placeholder(self):
		#encoder_image_inputs is a max_context_len sized list of [batch_size * image_rep_size] tensors
		self.encoder_image_inputs = [tf.placeholder(tf.float32,[None, image_size], name="encoder_image_inputs") for i in range(max_context_len)]  
		#
		#encoder_text_inputs is a max_context_len sized list of [batch_size, max_utter_len] tensors
		self.encoder_text_inputs = [tf.placeholder(tf.int32,[None, max_utter_len], name="encoder_text_inputs") for j in range(max_context_len)] 
		#
		#pos_images is [batch_size, 1, image_size] tensor
		self.pos_images = tf.placeholder(tf.float32, [None, 1, image_size], name="pos_images")
		#
		#negs_images is [batch_size, num_neg_images, image_size] tensor
		self.negs_images = tf.placeholder(tf.float32, [None, num_neg_images, image_size], name="negs_images")



		
	#Input - [batch_size, max_utter_len]
	#Output - [batch_size, cell_state_size]
	def sentence_encoder(self, encoder_text_input, reuse_state):

		#Convert [batch_size, max_utter_len] into [batch_size, max_utter_len, word_embedding_size]
		# Look up embedding:
		# encoder_text_input: [batch_size, max_time]
		# encoder_embedded_input: [batch_size, max_time, embedding_size]
		
		#RNN cell converts [batch_size, max_utter_len, word_embedding_size] to [batch_size, cell_state_size]
		# Run Dynamic RNN
		# sentence_output: [batch_size, max_utter_len, cell_state_size]
		# sentence_state: [batch_size, cell_state_size]
		with tf.variable_scope('enc1', reuse = reuse_state):
			encoder_embedded_input = tf.nn.embedding_lookup(self.embedder, encoder_text_input)
		
			self.cell_enc1 = tf.nn.rnn_cell.GRUCell(cell_state_size)

			sentence_output, sentence_state = tf.nn.dynamic_rnn(self.cell_enc1, encoder_embedded_input,
		 									time_major=False, dtype=tf.float32)
		return sentence_state




	
	def image_encoder(self, encoder_image_inputs):
		#Input - [batch_size, image_size]
		#Output - [batch_size, image_embedding_size]
		with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):
			encoder_image_states = tf.contrib.layers.fully_connected(encoder_image_inputs, 
												image_embedding_size, 
												activation_fn = None)
		return encoder_image_states
		

		

	#Encode the context. To be passed to decoder initial state
	def context_encoder(self, encoder_conc_input):
		#encoder_context__input is max_context_len * [batch_size, cell_state_size + image_embedding_size] tensors
		
		#RNN cell converts [batch_size, max_context_len, cell_state_size + image_embedding_size] to [batch_size, max_context_len, cell_state_size] and returns
		# Run Dynamic RNN
		# context_output: [batch_size, max_context_len, cell_state_size + image_embedding_size]
		# context_state: [batch_size, max_context_len, cell_state_size]
		with tf.variable_scope('enc2'):
			self.cell_enc2 = tf.nn.rnn_cell.GRUCell(cell_state_size)
			context_output, context_state = tf.nn.dynamic_rnn(self.cell_enc2, encoder_conc_input,
		 												time_major=False, dtype=tf.float32)
		return context_output, context_state




	def concat_and_transpose(self, encoder_text_states, encoder_image_states):
		#encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		#encoder_image_states is max_context_len * [batch_size, cell_state_size] tensors		
		
		text_image_states = []
		for i in range(max_context_len):
			text_image_states.append(tf.concat([encoder_text_states[i], encoder_image_states[i]], 1))		
		#text_image_states is max_context_len * [batch_size, cell_state_size + image_embedding_size]

		#Convert text_image_states from list of 2D tensors to a 3D tensor
		#Replace text_image_states with encoder_text_states in below line for Text Only Task
		encoder_conc_states = tf.stack(text_image_states, axis=1)
		#encoder_conc_states is [batch_size, max_context_len, cell_state_size + image_embedding_size] tensor
		
		return encoder_conc_states

	






	#Encode each utterance. Append all utterance states. 
	def hierarchical_encoder(self):		
		#encoder_text_inputs is max_context_len * [batch_size, max_utter_len] tensors
		#encoder_image_inputs is max_context_len * [batch_size, image_size] tensors
		
		reuse_state = False
		encoder_text_states = []
		for encoder_text_input in self.encoder_text_inputs:
			encoder_text_state = self.sentence_encoder(encoder_text_input, reuse_state)
			encoder_text_states.append(encoder_text_state)
			reuse_state = True
		#encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		
		#with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):

		encoder_image_states = []
		for encoder_image_input in self.encoder_image_inputs:			
			encoder_image_state = self.image_encoder(encoder_image_input)
			encoder_image_states.append(encoder_image_state)
		#encoder_image_states is max_context_len * [batch_size, cell_state_size] tensors

		encoder_conc_states = self.concat_and_transpose(encoder_text_states, encoder_image_states)
		#enc_concat_text_img_states is [batch_size, max_context_len, cell_state_size + image_embedding_size] tensor
		
		context_output, context_state = self.context_encoder(encoder_conc_states)
		#context_state is [batch_size, cell_state_size] tensor
		
		return context_output, context_state



	def cosine_similarity(self, image_set_1, image_set_2):
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

	

	def loss(self, context_state):
		#context_state is [batch_size, image_embedding_size]
		#pos_images is [batch_size, 1, image_size]

		# with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):

		pos_images = tf.reshape(self.pos_images, [batch_size, image_size])
		#pos_images is now [batch_size, image_size]
		encoded_pos_images = self.image_encoder(pos_images)
		#encoded_pos_images is [batch_size, image_embedding_size]
		#cell_state_size = image_embedding_size	
		cosine_sim_pos = self.cosine_similarity(context_state, encoded_pos_images)
		#cosine_sim_pos is of [batch_size]

		negs_images_reshaped = tf.transpose(self.negs_images, [1, 0, 2])
		#negs_image_reshaped is [num_neg_images, batch_size, image_size]
		encoded_negs_images = []
		#encoded_negs_images is [num_neg_images, batch_size, cell_state_size]
		cosine_sim_negs = []
		#cosine_sim_negs is [num_neg_images, batch_size]

		losses = []
		act_losses = []
		ones = tf.ones([batch_size])
		zeros = tf.zeros([batch_size])

		for neg_images_reshaped in tf.unstack(negs_images_reshaped):

			encoded_neg_images = self.image_encoder(neg_images_reshaped)
			#encoded_neg_images is [batch_size, cell_state_size]
			#cell_state_size = image_embedding_size

			cosine_sim_neg = self.cosine_similarity(context_state, encoded_neg_images)
			encoded_negs_images.append(encoded_neg_images)
			cosine_sim_negs.append(cosine_sim_neg)
			#cosine_sim_neg is of [batch_size]

			act_loss = ones - cosine_sim_pos + cosine_sim_neg
			per_loss = tf.maximum(zeros, act_loss)
			losses.append(per_loss)
			act_losses.append(act_loss)	
	
			loss = tf.reduce_sum(tf.add_n(losses))

   		return loss, losses, act_losses, context_state, pos_images, encoded_pos_images, cosine_sim_pos, negs_images_reshaped, encoded_negs_images, cosine_sim_negs




	def train(self, losses):

		parameters=tf.trainable_variables()    
		optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
		global_step=tf.Variable(0,name="global_step",trainable='False')

		#No gradient clipping
		#train_op=optimizer.minimize(losses,global_step=global_step)

		#With gradient clipping
		gradients=tf.gradients(losses,parameters)		
		clipped_gradients,norm=tf.clip_by_global_norm(gradients, max_gradient_norm)
		train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)

		return train_op, clipped_gradients




	def sim(self, context_state):
		#context_state is [batch_size, image_embedding_size]
		#pos_images is [batch_size, image_size]
		
		#with tf.variable_scope('img_enc', reuse=tf.AUTO_REUSE):
			
		encoded_pos_images = self.image_encoder(self.pos_images)
		#encoded_pos_images is [batch_size, image_embedding_size]
		#cell_state_size = image_embedding_size	

		pos_image_reshaped = tf.reshape(encoded_pos_images, [batch_size, cell_state_size])
		cosine_sim_pos = self.cosine_similarity(context_state, pos_image_reshaped)
		#pos_image_reshaped is [batch_size, cell_state_size]
		#cosine_sim_pos is of [batch_size]

		negs_image_reshaped = tf.transpose(self.negs_images, [1, 0, 2])
		#negs_image_reshaped is [num_neg_images, batch_size, image_size]
		const = tf.ones([batch_size])
		zeros = tf.zeros([batch_size])
		cosine_sim_negs = []
		for image in tf.unstack(negs_image_reshaped):
			
			encoded_image = self.image_encoder(image)
			#encoded_image is [batch_size, cell_state_size]
			#cell_state_size = image_embedding_size

			cosine_sim_neg = self.cosine_similarity(context_state, encoded_image)
			#cosine_sim_neg is of [batch_size]
			cosine_sim_negs.append(cosine_sim_neg)

		#cosine_sim_neg is list of num_neg_images * [batch_size]
		cosine_sim_neg_tensor = tf.stack(cosine_sim_negs)
		#cosine_sim_neg_tensor is [num_neg_images, batch_size]
		cosine_sim_neg_tensor_transposed = tf.transpose(cosine_sim_neg_tensor, [1,0])
		#cosine_sim_neg_tensor_transposed is [batch_size, num_neg_images]

   		return cosine_sim_pos, cosine_sim_neg_tensor_transposed

















