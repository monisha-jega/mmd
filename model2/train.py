from model import *
from util import *
import random, pickle, sys

#Load data
train_data = pickle.load(open(train_data_file, "rb"))[:]
val_data = pickle.load(open(val_data_file, "rb"))[:]
#test_data = pickle.load(open(val_data_file, "rb"))
#Load vocab
vocab = pickle.load(open(vocab_word_to_id_file, "rb"))



#MAKE GRAPH
final_features = hierarchical_and_concat()   
output = MLP(final_features)
lossout, train_op = loss_train(output)
sim = sim_func(output)
saver = saver()


#Initializations
n_batches = len(train_data)/batch_size
nv_batches = len(val_data)/batch_size

#nt_batches = len(test_data)/batch_size
best_val_loss = float("inf")

#RUN
with tf.Session() as sess:
	if restore_trained == True:
		saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
		print("model loaded")
	else:
		sess.run(tf.global_variables_initializer())

	for e in range(1, epochs+1):
		print("Epoch %d" %(e))
		random.shuffle(train_data)
		train_loss = 0

		for b in range(n_batches):
			#Get batch
			train_data_batch=train_data[b*batch_size:(b+1)*batch_size]
			feeding_dict = process_batch(train_data_batch)
			#Train
			batch_loss ,_ = sess.run([lossout, train_op], feed_dict=feeding_dict)        
			batch_loss = np.sum(batch_loss)
			per_loss = batch_loss/float(batch_size)    
			print('Epoch  %d Batch %d  point loss (avg over batch) =%.6f' %(e, b, per_loss))
			train_loss = train_loss + batch_loss
			avg_train_loss = float(train_loss)/float(b+1)            
		
		#Print train loss after each epoch
		per_train_loss = train_loss/float(len(train_data))
		print('Epoch %d completed with loss : %6f' %(e, per_train_loss))           
		print(nv_batches)

		#Validation
		val_loss = 0
		val_ranks_all = []
		for b in range(nv_batches):
			print b+1
			#Get batch
			val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
			feeding_dict = process_batch(val_data_batch)
			#
			loss, (val_pos_sim, val_negs_sim) = sess.run([lossout, sim], feed_dict=feeding_dict)
			#Loss
			batch_loss = np.sum(loss)
			val_loss = val_loss + batch_loss 
			#Ranks
			val_ranks_all += get_ranks(val_pos_sim, val_negs_sim)
		#Print val loss
		val_loss = val_loss/float(len(val_data))           
		print 'Val loss ', val_loss 
		for m in [1,2,3,4,5]:
			print 'Val recall @ ', m, ' : ', recall(val_ranks_all, m)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			#Save trained model
			saver.save(sess, model_dump_dir + str(e)) 

		
	
