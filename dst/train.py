import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parameters import *
from model import *
import pickle, random, math
import numpy as np

#Load data
train_data = pickle.load(open(data_dump_dir + "train_dst.pkl"))[:20]
val_data = pickle.load(open(data_dump_dir + "train_dst.pkl"))[:5]

#Calculate no of batches
n_batches = int(math.ceil(len(train_data)/float(batch_size)))
nv_batches = int(math.ceil(len(val_data)/float(batch_size)))


best_val_loss = float("inf")
#Run training
with tf.Session() as sess:	
	
	if restore_trained == True:
		saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
		print("model loaded")
	else:
		sess.run(tf.global_variables_initializer())
		
	
	for e in range(1, epochs+1):
		#Shuffle at each epoch
		random.shuffle(train_data)

		train_loss=0
		for b in range(n_batches):
			print 'Step ', b+1

			#Get batch
			train_data_batch = train_data[b*batch_size:(b+1)*batch_size]
			text_batch, targets_batch = zip(*train_data_batch)
			#
			#Train
			loss, op = sess.run([loss_node, train_op], feed_dict={text_inputs : np.array(text_batch), 
																	  targets : np.array(targets_batch)
																	  })
			#
			#Loss
			batch_loss = np.sum(loss)
			per_loss = batch_loss/float(batch_size) 
			print('Epoch  %d Batch %d train loss (avg over batch) =%.6f' %(e, b, per_loss))
			train_loss += batch_loss           
		
		#Print train loss after each epoch
		epoch_train_loss = train_loss/float(len(train_data))             
		print('Epoch %d completed' %(e))
		print 'Train loss ', epoch_train_loss 


		#Validation
		val_loss = 0
		targets_all = []
		preds_all = []
		for b in range(nv_batches):			
			#Get batch
			val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
			text_batch, targets_batch = zip(*val_data_batch)
			targets_all += targets_batch
			#
			#Validate
			preds, loss = sess.run([output_layer, loss_node], feed_dict={text_inputs : np.array(text_batch), 
																		  targets : np.array(targets_batch)
																		  })
			preds_all += list(np.argmax(preds,1))
			#
			#Loss
			batch_loss = np.sum(loss)
			val_loss += batch_loss 

		#Print val loss
		val_loss = val_loss/float(len(val_data)) 
		print 'Val loss ', val_loss         
		print("Val accuracy ", accuracy_score(targets_all, preds_all))
		print("Val precision ", precision_score(targets_all, preds_all, average = 'micro'))
		print("Val recall ", recall_score(targets_all, preds_all, average = 'micro'))
		print("Val f-score ", f1_score(targets_all, preds_all, average = 'micro'))
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			#Save trained model if val loss is less than nest loss
			saver.save(sess, model_dump_dir+'model_' + str(e)) 
