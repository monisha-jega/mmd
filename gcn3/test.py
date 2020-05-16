from parameters import *
from GCN import *
from batch_util import *
import pickle, math, random


#Load test data
test_data = pickle.load(open(data_dump_dir+"test_binarized_data.pkl", 'rb'))[:10000]
print("Loaded test data")
#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_id_to_word.pkl","rb"))
vocab_size = len(vocab)

#Calculate number of batches
nt_batches = int(math.ceil(len(test_data)/float(batch_size)))


#Create graph
graph1 = tf.Graph()
with graph1.as_default():

	model = HREI(vocab_size)   
	model.create_placeholder()
	#
	model.out, model.code = model.hierarchical_encoder()
	#	
	model.losses = model.loss(model.code)
	#	 
	model.simout = model.sim(model.code) 
	#
	model.saver = tf.train.Saver()
print("tf graph made")

#Run test 
test_loss = 0
test_ranks_all = []
with tf.Session(graph=graph1) as sess:

	#Restore pre-trained model
	model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
	print("Loaded model")

	for b in range(nt_batches):
		print 'Step ', b+1

		#Get batch
		test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
		feed_dict = process_batch(model, test_data_batch)
		#
		#Test
		loss, (test_pos_sim, test_negs_sim) = sess.run([model.losses, model.simout], feed_dict=feed_dict)
		#
		#Loss
		batch_loss = np.sum(loss)
		test_loss += batch_loss    
		#Ranks
		test_ranks_all += get_ranks(test_pos_sim, test_negs_sim)        
		# for m in [1,2,3]:
		#     print 'Intermediate Test recall@ ', str(m), " : ", recall(test_ranks_all, m)   

	#Print test loss
	test_loss = test_loss/float(len(test_data))             
	print 'Test loss ', test_loss  
	for m in [1,2,3]:
		print 'Test recall@ ', str(m), " : ", recall(test_ranks_all, m)   