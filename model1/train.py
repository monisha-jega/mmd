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
loss, train_op, flosses, tl, ol = loss_func(output)
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
			feeding_dict, _, _ = process_batch(train_data_batch)
			#Train
			o, f, tlp, olp, batch_loss ,_ = sess.run([output, flosses, tl, ol, loss, train_op], feed_dict=feeding_dict)        
			#Print Loss
			#print(f, len(f))
			#print(tlp, olp)
			batch_loss = np.sum(batch_loss)
			per_loss = batch_loss/float(batch_size)    
			print('Epoch  %d Batch %d  point loss (avg over batch) =%.6f' %(e, b, per_loss))
			train_loss = train_loss + batch_loss
			avg_train_loss = float(train_loss)/float(b+1)            
		
		#Print train loss after each epoch
		per_train_loss = train_loss/float(len(train_data))
		print('Epoch %d completed with loss : %6f' %(e, per_train_loss))           


		#Validation
		val_loss = 0
		pos_images = []
		neg_images = []
		preds = []
		print(nv_batches)
		for b in range(nv_batches):
			#print("Val batch", b+1)
			#Get batch
			val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
			feeding_dict, batch_pos_image, batch_neg_images = process_batch(val_data_batch)
			pos_images += batch_pos_image
			neg_images += batch_neg_images
			#Loss
			output_image_rep_preds, batch_loss = sess.run([output, loss], feed_dict = feeding_dict)
			#output_image_rep_preds is num_features * [batch_size, feature_size] 
			output_image_rep_preds_reshaped = []
			for i in range(batch_size):
				output_image_rep_preds_reshaped.append([[output_image_rep_preds[j][i][k] for k in range(feature_sizes[j])] for j in range(num_features)]) 
			#output_image_rep_preds_reshaped batch_size * num_features * feature_size
			preds += output_image_rep_preds_reshaped
			#Loss
			batch_loss = np.sum(batch_loss)
			val_loss = val_loss + batch_loss 
		#Print val loss
		val_loss = val_loss/float(len(val_data)) 
		print 'Val loss ', val_loss 

		#Calculate accuracy of image retrieval using output_image_rep_pred
		ranks = rank_images(preds, pos_images, neg_images)
		recall1, recall2, recall3, recall4, recall5 = recall_m(ranks, 1), recall_m(ranks, 2), recall_m(ranks, 3), recall_m(ranks, 4), recall_m(ranks, 5)
		print("Recall @1, 2, 3, 4, 5 : %.3f, %.3f, %.3f, %.3f, %.3f" %(recall1, recall2, recall3, recall4, recall5))
	   
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			#Save trained model
			saver.save(sess, model_dump_dir+'model_' + str(e)) 

		
	
