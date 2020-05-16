from parameters import *
from GCN import *
from batch_util import *
import pickle, math, random


#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"0_train_binarized_data.pkl", 'rb'))[:]
val_data = pickle.load(open(data_dump_dir+"test_binarized_data.pkl", 'rb'))[:]
print("data loaded", len(train_data), len(val_data))
#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_word_to_id.pkl","rb"))
vocab_size = len(vocab)

#Calculate number of batches
n_batches = len(train_data)/batch_size
nv_batches = len(val_data)/batch_size


#Create graph
graph1 = tf.Graph()
with graph1.as_default():
	model = HREI(vocab_size)   
	model.create_placeholder()
	#
	model.code = model.hierarchical_encoder()
	#
	model.losses = model.loss(model.code)
	model.train_op, _ = model.train(model.losses)
	#
	model.simout = model.sim(model.code)   
	#
	model.saver = tf.train.Saver()
print("tf graph made")


best_val_loss = float("inf")
#Run training
with tf.Session(graph=graph1) as sess:

	if restore_trained == True:
		model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
		print("model loaded")
	else:
		sess.run(tf.global_variables_initializer())	
		
	for e in range(1, epochs+1):
		#Shuffle data at each epoch
		random.shuffle(train_data)

		train_loss = 0
		train_ranks_all = []
		for b in range(n_batches):
			print 'Step ', b+1  

			if b == n_batches-1:
				model.saver.save(sess, model_dump_dir+'model_' + str(e))          

			#Get batch
			train_data_batch=train_data[b*batch_size:(b+1)*batch_size]
			feed_dict = process_batch(model, train_data_batch)
			#
			#Train
			loss, (train_pos_sim, train_negs_sim), _ = sess.run([model.losses, model.simout, model.train_op], feed_dict=feed_dict)
			#
			#Loss
			batch_loss = np.sum(loss)
			per_loss = batch_loss/float(batch_size)    
			print('Epoch  %d Batch %d train loss (avg over batch) =%.6f' %(e, b, per_loss))
			train_loss += batch_loss
			avg_train_loss = float(train_loss)/float(b+1) 
			#Rank
			train_ranks_all += get_ranks(train_pos_sim, train_negs_sim)       
			# print("Intermediate Ranks", train_ranks_all) 
			# for m in [1,2,3]:
			#     print 'Intermediate Train recall @ ', m, ' : ', recall(train_ranks_all, m)


		#Print train loss after each epoch
		print('Epoch %d completed' %(e))
		epoch_train_loss = train_loss/float(len(train_data))             
		print 'Train loss ', epoch_train_loss
		#print(train_ranks_all) 
		for m in [1,2,3,4,5]:
			print 'Train recall @ ', m, ' : ', recall(train_ranks_all, m)

		#Validation at the end of each epoch
		val_loss = 0
		val_ranks_all = []	
		for b in range(nv_batches):
			#Get batch
			val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
			feed_dict = process_batch(model, val_data_batch)
			#
			#Test
			loss, (val_pos_sim, val_negs_sim) = sess.run([model.losses, model.simout], feed_dict=feed_dict)
			#
			#Loss
			batch_loss = np.sum(loss)
			val_loss += batch_loss 
			#Ranks
			val_ranks_all += get_ranks(val_pos_sim, val_negs_sim)

		#Print val loss
		val_loss = val_loss/float(len(val_data))           
		print 'Val loss ', val_loss 
		for m in [1,2,3]:
			print 'Val recall @ ', m, ' : ', recall(val_ranks_all, m)
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			#Save trained model if val loss is lessser than best val loss
			model.saver.save(sess, model_dump_dir+'model_' + str(e)) 

	print("Saved")
	#Validation at the end of each epoch
	val_loss = 0
	val_ranks_all = []	
	for b in range(nv_batches):
		#Get batch
		val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
		feed_dict = process_batch(model, val_data_batch)
		#
		#Test
		loss, (val_pos_sim, val_negs_sim) = sess.run([model.losses, model.simout], feed_dict=feed_dict)
		#
		#Loss
		batch_loss = np.sum(loss)
		val_loss += batch_loss 
		#Ranks
		val_ranks_all += get_ranks(val_pos_sim, val_negs_sim)

	#Print val loss
	val_loss = val_loss/float(len(val_data))           
	print 'Val loss ', val_loss 
	for m in [1,2,3]:
		print 'Val recall @ ', m, ' : ', recall(val_ranks_all, m)
