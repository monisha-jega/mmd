from parameters import *
from HRED import *
from batch_util import *
import pickle, math, random
import nltk
import nltk.translate.nist_score

#Load test data
test_data = pickle.load(open(data_dump_dir+"test_binarized_data.pkl", 'rb'))[:1000]

#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_id_to_word.pkl","rb"))
vocab_size = len(vocab)

#Load number of batches
nt_batches = int(math.ceil(len(test_data)/float(batch_size)))

#Create graph
graph1 = tf.Graph()
with graph1.as_default():

	model = HRED(vocab_size)   
	model.create_placeholder()
	#
	model.out, model.code = model.hierarchical_encoder()
	model.context_for_dec = model.decoder_cell_and_initial_state(model.out, model.code)
	#
	model.logits = model.decoder(model.context_for_dec)
	model.losses = model.loss(model.logits)
	model.train_op, _ = model.train(model.losses)
	#
	model.GTlogits = model.GTdecoder(model.context_for_dec)
	model.GTlosses = model.GTloss(model.GTlogits)
	model.GTtrain_op, _ = model.GTtrain(model.GTlosses)
	#
	model.output = model.inference_decoder(model.context_for_dec)
	#
	model.saver = tf.train.Saver()
print("tf graph made")


#Run test 
f = open(data_dump_dir+'test_outputs', 'w') 
test_loss = 0
with tf.Session(graph=graph1) as sess:
	
	#Restore pre-trained model
	model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
	print('model loaded')

	for b in range(nt_batches):
		print 'Step ', b+1

		#Get batch
		test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
		feed_dict = process_batch(model, test_data_batch)
		#
		#Test
		loss, reply = sess.run([model.GTlosses, model.output], feed_dict=feed_dict)
		# print(dec[0][0], reply[0][0], inf[0][0], tar[0][0])
		# print(dec[0][1], reply[0][1], inf[0][1], tar[0][1])
		# print(dec[0][2], reply[0][2], inf[0][2], tar[0][2])
		# print(logits[0][0], targs[0][0], reply[0][0], probs[0][0])
		# print(logits[0][1], targs[0][1], reply[0][1], probs[0][1])
		# print(logits[0][2], targs[0][2], reply[0][2], probs[0][2])
		#
		#Loss
		batch_loss = np.sum(loss)
		test_loss += batch_loss  
		#Write test output
		for each in reply:
			words = ""
			for word_ids in each:
				if use_beam == 0:
					#For greedy decoding
					word = vocab[word_ids]
				else:
					#For beam search
					word = vocab[word_ids[0]]				
				
				words += word + " "
			f.write(words + "\n") 
	#Print test loss
	test_loss = test_loss/float(len(test_data))             
	print 'Test loss ', test_loss     

f.close()   
		
		



#Targets
targets_ = open(data_dump_dir + 'test_targets').readlines()[:1000]
targets = []
for sent in targets_:
	try:
		sent = nltk.word_tokenize(sent)
	except:
		sent = sent.split(" ")
	targets.append([sent]) 


#Outputs
outputs_ = open(data_dump_dir + 'test_outputs').readlines()[:len(test_data)]
outputs = []
for sent in outputs_:
	try:
		sent = nltk.word_tokenize(sent)
	except:
		sent = sent.split(" ")
	outputs.append(sent) 

#Get BLEU score between targets and outputs
print "BLEU : ", nltk.translate.bleu_score.corpus_bleu(targets, outputs)
#Get NIST score between targets and outputs
print "NIST : ", nltk.translate.nist_score.corpus_nist(targets, outputs)
