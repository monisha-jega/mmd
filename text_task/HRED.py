import tensorflow as tf
from parameters import *
import pickle
from tensorflow.python.layers import core as layers_core

#Get embeddings of vocab
embeddings_google = pickle.load(open(data_dump_dir + 'vocab_embeddings.pkl', 'rb'))

class HRED():

	def __init__(self, vocab_size):
		self.vocab_size = vocab_size

		#Convert vocab embeddings into tensor
		self.embedding_matrix = tf.constant(embeddings_google, name= "embedding_matrix")
		self.embedder = tf.get_variable("embedding_matrix", [self.vocab_size, word_embedding_size])

		self.cell_dec = tf.nn.rnn_cell.GRUCell(cell_state_size)
		



	def create_placeholder(self):
		#encoder_image_inputs is a max_context_len sized list of [batch_size * image_rep_size] tensors
		self.encoder_image_inputs = [tf.placeholder(tf.float32,[None, image_size], name="encoder_image_inputs") for i in range(max_context_len)]  
		#
		#encoder_text_inputs is a max_context_len sized list of [batch_size, max_utter_len] tensors
		self.encoder_text_inputs = [tf.placeholder(tf.int32,[None, max_utter_len], name="encoder_text_inputs") for j in range(max_context_len)] 
		#
		#decoder_text_input is [batch_size, max_utter_len] tensor
		self.decoder_inputs = tf.placeholder(tf.int32,[None, max_utter_len], name="decoder_inputs")
		#
		#target is [batch_size, max_utter_len] tensor
		self.targets = tf.placeholder(tf.int32,[None, max_utter_len], name="targets")



		
	def sentence_encoder(self, encoder_text_input, reuse_state):
		# Look up embedding:
		# encoder_text_input: [batch_size, max_time]
		encoder_embedded_input = tf.nn.embedding_lookup(self.embedder, encoder_text_input)
		# encoder_embedded_input: [batch_size, max_time, embedding_size]
			
		with tf.variable_scope('enc1', reuse = reuse_state):
			#RNN cell converts [batch_size, max_utter_len, word_embedding_size] to [batch_size, cell_state_size]
			self.cell_enc1_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
			self.cell_enc1_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)

			# Run Dynamic RNN
			(sentence_output_1, sentence_output_2), (sentence_state_1, sentence_state_2) = \
			tf.nn.bidirectional_dynamic_rnn(self.cell_enc1_fw, self.cell_enc1_bw, encoder_embedded_input,
		 									time_major=False, dtype=tf.float32)

		sentence_state_stack = tf.stack([sentence_state_1, sentence_state_2])
		sentence_state_mean = tf.reduce_mean(sentence_state_stack,0)
		# sentence_output: [batch_size, max_utter_len, cell_state_size]
		# sentence_state: [batch_size, cell_state_size]		
		return sentence_state_mean




	
	def image_encoder(self, encoder_image_inputs):
		#encoder_image_inputs - [batch_size, image_size]		
		encoder_image_states = tf.contrib.layers.fully_connected(encoder_image_inputs, 
												image_embedding_size, 
												activation_fn = None)
		#encoder_image_states - [batch_size, image_embedding_size]
		return encoder_image_states



		

	#Encode the context. To be passed to decoder initial state
	def context_encoder(self, encoder_conc_input):
		with tf.variable_scope('enc2'):
			#Bidirectional RNN cell converts [batch_size, max_context_len, cell_state_size + image_embedding_size] to [batch_size, max_context_len, cell_state_size] and returns
			self.cell_enc2_fw = tf.nn.rnn_cell.GRUCell(cell_state_size)
			self.cell_enc2_bw = tf.nn.rnn_cell.GRUCell(cell_state_size)

			# Run Dynamic RNN
			#encoder_conc_input is [batch_size, max_context_len, cell_state_size + image_embeddings_size]
			(context_output_1, context_output_2), (context_state_1, context_state_2) = \
			tf.nn.bidirectional_dynamic_rnn(self.cell_enc2_fw, self.cell_enc2_bw, encoder_conc_input,
		 												time_major=False, dtype=tf.float32)

		context_state_stack = tf.stack([context_state_1, context_state_2])
		context_state_mean = tf.reduce_mean(context_state_stack,0)
		context_output = tf.concat([context_output_1, context_output_2], 2)
		# context_output_1, context_output_2, context_output: [batch_size, max_context_len, cell_state_size]
		# context_state_1, context_state_2, context_state: [batch_size, cell_state_size]		
		return context_output, context_state_mean




	def concat_and_transpose(self, encoder_text_states, encoder_image_states):
		#encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		#encoder_image_states is max_context_len * [batch_size, cell_state_size] tensors		
		
		text_image_states = []
		for i in range(max_context_len):
			text_image_states.append(tf.concat([encoder_text_states[i], encoder_image_states[i]], 1))		
		#text_image_states is max_context_len * [batch_size, cell_state_size + image_embedding_size]

		#Convert text_image_states from list of 2D tensors to a 3D tensor
		if text_only == True:
			encoder_conc_states = tf.stack(encoder_text_states, axis=1)
		else:
			encoder_conc_states = tf.stack(text_image_states, axis=1)
		#encoder_conc_states is [batch_size, max_context_len, cell_state_size + image_embedding_size]	
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
		# encoder_text_states is max_context_len * [batch_size, cell_state_size] tensors
		
		encoder_image_states = []
		for encoder_image_input in self.encoder_image_inputs:			
			encoder_image_state = self.image_encoder(encoder_image_input)
			encoder_image_states.append(encoder_image_state)
		# encoder_image_states is max_context_len * [batch_size, cell_state_size] tensors

		encoder_conc_states = self.concat_and_transpose(encoder_text_states, encoder_image_states)
		# enc_conc_states is [batch_size, max_context_len, cell_state_size + image_embedding_size] tensor
		
		context_output, context_state = self.context_encoder(encoder_conc_states)
		# context_output: [batch_size, max_context_len, cell_state_size]
		# context_state is [batch_size, cell_state_size] tensor		
		return context_output, context_state



	def output_projection(self):
		#Projection Layer after decoder
		projection_layer = layers_core.Dense(self.vocab_size, use_bias=False)
		return projection_layer



	#Create decoder cell, wrap in attention if required, and tile decoder initial state for attention (if being used)
	def decoder_cell_and_initial_state(self, context_output, context_state):
		with tf.variable_scope('dec'):
			if attention == True:
				# Create an attention mechanism
		  		attention_mechanism = tf.contrib.seq2seq.LuongAttention(cell_state_size, context_output, memory_sequence_length=None)
		  		#Decoder cell is wrapped in the attention mechanism above
				self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.cell_dec, attention_mechanism, 
		  													attention_layer_size=2*cell_state_size)
	  			decoder_initial_state = self.decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=context_state)
	  		else:
	  			self.decoder_cell = self.cell_dec
	  			decoder_initial_state = context_state
		return decoder_initial_state



	def decoder(self, decoder_initial_state):
		with tf.variable_scope('dec')
	  		#decoder_inputs is [batch_size, max_utter_len]
			decoder_embedded_inputs = tf.nn.embedding_lookup(self.embedder, self.decoder_inputs)
			#decoder_embedded_inputs: [batch_size, max_utter_len, embedding_size]
			#Decoder with helper
			helper = tf.contrib.seq2seq.TrainingHelper(# decoder_length: [batch_size] (All values = max_utter_len)
					decoder_embedded_inputs, tf.fill([batch_size], max_utter_len), time_major=False)			
			decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, decoder_initial_state,
			    									output_layer=self.output_projection())

			# Dynamic decoding
			final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
			# final_outputs.rnn_output: [batch_size, max_utter_length, vocab_size], list of RNN state.
			# final_outputs.sample_id: [batch_size, max_utter_length], list of argmax of rnn_output.
			# final_state: [batch_size, cell_state_size], list of final state of RNN on decode process.
			# final_sequence_lengths: [batch_size], list of each decoded sequence.
			logits = final_outputs.rnn_output
			return logits





	def GTdecoder(self, decoder_initial_state):
		with tf.variable_scope('dec', reuse=tf.AUTO_REUSE):	
			# Greedy Decoder with helper
			GThelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedder,
			    tf.fill([batch_size], start_word_id), end_word_id)
			GT_decoder = tf.contrib.seq2seq.BasicDecoder(
			    self.decoder_cell, GThelper, decoder_initial_state, output_layer=self.output_projection())

			# Dynamic decoding
			max_iters = tf.round(tf.reduce_max(max_utter_len))
			final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(GT_decoder, maximum_iterations=max_iters)
			# final_outputs.rnn_output: [batch_size, max_utter_length, vocab_size], list of RNN state.
			# final_outputs.sample_id: [batch_size, max_utter_length], list of argmax of rnn_output.
			# final_state: [batch_size, cell_state_size], list of final state of RNN on decode process.
			# final_sequence_lengths: [batch_size], list of each decoded sequence.
			logits = final_outputs.rnn_output
			return logits			
			
			# #Is this padding. Required?
			# to_pad = max_utter_len-tf.shape(outputs.rnn_output)[1]
			# one_hot_pad = tf.one_hot(pad_word_id, self.vocab_size)
			# tile_across_utterlen = tf.tile(one_hot_pad, [to_pad])
			# tile_across_utterlen_ = tf.reshape(tile_across_utterlen, [to_pad, self.vocab_size])
			# tile_across_batchsize = tf.tile(tile_across_utterlen_, [batch_size, 1])
			# filler = tf.reshape(tile_across_batchsize, [batch_size, to_pad, self.vocab_size])
			# # filler = tf.fill([batch_size, max_utter_len-tf.shape(outputs.rnn_output)[1], self.vocab_size], )
			# filler = tf.cast(filler, tf.float32)
			# filled = tf.concat([outputs.rnn_output, filler], 1)
			# # return filled, outputs.sample_id
			# return filled





	def loss(self, logits):
		#labels is [batch_size, decoder_length] (NOT ONE-HOT)
		#logits is [batch_size, decoder_length, vocab_size]
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits)
		#loss is [batch_size, decoder_length]
		return loss


	def GTloss(self, logits):
		#labels is [batch_size, decoder_length] (NOT ONE-HOT)
		#logits is [batch_size, decoder_length, vocab_size]
		with tf.variable_scope('GT'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits)
		#loss is [batch_size, decoder_length]
		return loss




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


	def GTtrain(self, losses):
		with tf.variable_scope('GT'):
			parameters=tf.trainable_variables()    
			optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08, name="AdamGT")
			global_step=tf.Variable(0,name="global_step",trainable='False')

			#No gradient clipping
			#train_op=optimizer.minimize(losses,global_step=global_step)
			#With gradient clipping
			gradients=tf.gradients(losses,parameters)		
			clipped_gradients,norm=tf.clip_by_global_norm(gradients, max_gradient_norm)
			train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)

		return train_op, clipped_gradients







	def inference_decoder(self, decoder_initial_state):	
		
		if use_beam_search == False:
			print("Using greedy decoder")
			#Greedy Decode
			with tf.variable_scope('dec', reuse=tf.AUTO_REUSE):		
				#Greedy decoder with helper
				infhelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedder,
				    tf.fill([batch_size], start_word_id), end_word_id)
				infdecoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, infhelper, decoder_initial_state,
				    output_layer=self.output_projection())

				# Dynamic decoding
				max_iters = tf.round(tf.reduce_max(max_utter_len))
				final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(infdecoder, maximum_iterations=max_iters)
				# final_outputs.rnn_output: [batch_size, max_utter_length, vocab_size], list of RNN state.
				# final_outputs.sample_id: [batch_size, max_utter_length], list of argmax of rnn_output.
				# final_state: [batch_size, cell_state_size], list of final state of RNN on decode process.
				# final_sequence_lengths: [batch_size], list of each decoded sequence.
				reply = final_outputs.sample_id

		else:
			# Beam Search	
			# Define a beam-search decoder
			with tf.variable_scope('dec', reuse=tf.AUTO_REUSE):
				#Tile decoder initial state by beam width
				decoder_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, multiplier=beam_width)
				
				beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
							        cell=self.decoder_cell,
							        embedding=self.embedder,
							        start_tokens=tf.fill([batch_size], start_word_id),
							        end_token=end_word_id,
							        initial_state=decoder_initial_state,
							        beam_width=beam_width,
							        output_layer=self.output_projection(),
							        length_penalty_weight=0.0)

				# Dynamic decoding
				beam_max_iters = tf.round(tf.reduce_max(max_utter_len))
				final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(beam_decoder, maximum_iterations=max_iters)
				# final_outputs.rnn_output: [batch_size, max_utter_length, vocab_size], list of RNN state.
				# final_outputs.sample_id: [batch_size, max_utter_length], list of argmax of rnn_output.
				# final_state: [batch_size, cell_state_size], list of final state of RNN on decode process.
				# final_sequence_lengths: [batch_size], list of each decoded sequence.
				reply = final_outputs.predicted_ids

				return reply









		



