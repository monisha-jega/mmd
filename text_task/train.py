from parameters import *
from HRED import *
from batch_util import *
import pickle, math, random


#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"train_binarized_data.pkl", 'rb'))
val_data = pickle.load(open(data_dump_dir+"val_binarized_data.pkl", 'rb'))
print("data loaded")
#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_word_to_id.pkl","rb"))
vocab_size = len(vocab)

#Calculate number of batches
n_batches = int(math.ceil(len(train_data)/float(batch_size)))
nv_batches = int(math.ceil(len(val_data)/float(batch_size)))


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


best_val_loss = float("inf")
#Run training
with tf.Session(graph=graph1) as sess:

    if restore_trained == True:
        model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
        print("model loaded")
    else:
        sess.run(tf.global_variables_initializer())
    

    for e in range(1, epochs+1):
        #Shuffle at each epoch
        random.shuffle(train_data)

        train_loss=0
        for b in range(n_batches):
            print 'Step ',b+1

            #Get batch
            train_data_batch=train_data[b*batch_size:(b+1)*batch_size]
            feed_dict = process_batch(model, train_data_batch)
            #Train
            if e < start_greedy_from_epoch:
                loss, logs, _ = sess.run([model.losses, model.logits, model.train_op], feed_dict=feed_dict)
            else:
                print("Training with greedy decoder")
                loss, logs, _ = sess.run([model.GTlosses, GTlogits, model.GTtrain_op], feed_dict=feed_dict)            
            #
            #Loss
            batch_loss = np.sum(loss)
            per_loss = batch_loss/float(batch_size)    
            print('Epoch  %d Batch %d (Step %d) train loss (avg over batch) =%.6f' %(e, b, step, per_loss))
            train_loss += batch_loss
            avg_train_loss = float(train_loss)/float(b+1)            
        
        #Print train loss after each epoch
        print('Epoch %d completed' %(e))
        epoch_train_loss = train_loss/float(len(train_data))             
        print 'Train loss ', epoch_train_loss 


        #Validation at the end of each epoch
        val_loss = 0        
        for b in range(nv_batches):
            #Get batch
            val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
            feed_dict = process_batch(model, val_data_batch)
            #Validate
            loss = sess.run(model.GTlosses, feed_dict=feed_dict)
            #Loss
            batch_loss = np.sum(loss)
            val_loss += batch_loss

        #Print val loss
        val_loss = val_loss/float(len(val_data)) 
        print 'Val loss ', val_loss 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #Save trained model if val loss is less than the best val loss
            model.saver.save(sess, model_dump_dir+'model_' + str(e)) 

        
    
