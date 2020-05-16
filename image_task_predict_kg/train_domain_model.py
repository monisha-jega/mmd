from parameters_domain_model import *
from batch_util_domain_model import *
import pickle, math, random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"train_kg.pkl", 'rb'))[:]
val_data = pickle.load(open(data_dump_dir+"val_kg.pkl", 'rb'))[:]
test_data = pickle.load(open(data_dump_dir+"test_kg.pkl", 'rb'))[:]
print("data loaded", len(train_data), len(val_data), len(test_data))
# #Load vocabulary
# vocab = pickle.load(open(data_dump_dir+"vocab_word_to_id.pkl","rb"))
# vocab_size = len(vocab)

#Load number of batches
n_batches = int(math.ceil(len(train_data)/float(batch_size)))

hidden_units = 100
gender_dim = 4

#image_inputs is [batch_size * image_rep_size]
image_inputs = tf.placeholder(tf.float32,[None, image_size], name="image_inputs") 
#gender_targets is [batch_size * 4]
gender_targets = tf.placeholder(tf.int32,[None], name="gender_target") 

#Input - [batch_size, image_size]
#Output - [batch_size, 4]
layer1 = tf.layers.dense(image_inputs, hidden_units, name="layer1")
output_layer = tf.layers.dense(layer1, gender_dim, name="output_layer")

#labels is [batch_size] (NOT ONE-HOT)
#logits is [batch_size,  4]
loss_node = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gender_targets, logits=output_layer))
optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op=optimizer.minimize(loss_node)

saver = tf.train.Saver()
print("tf graph created")



best_val_loss = float("inf")
#Run training
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    if restore_trained == True:
        saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
        print("model loaded")
    
    for e in range(1, epochs+1):
        #random.shuffle(train_data)
        train_loss=0
        step = 0
        
        for b in range(n_batches):
            step += 1
            print 'Step ', step

            #Get batch
            train_data_batch=train_data[b*batch_size:(b+1)*batch_size]
            images, targets = process_batch(train_data_batch)
            #print(images, targets)
            
            loss, op = sess.run([loss_node, train_op], feed_dict={image_inputs : np.array(images), gender_targets : np.array(targets)})
            #Loss
            batch_loss = np.sum(loss)
            per_loss = batch_loss/float(batch_size)    
            print('Epoch  %d Batch %d (Step %d) train loss (avg over batch) =%.6f' %(e, b, step, per_loss))
            train_loss = train_loss + batch_loss
            avg_train_loss = float(train_loss)/float(b+1)            
        
        #Print train loss after each epoch
        print('Epoch %d completed' %(e))
        epoch_train_loss = train_loss/float(len(train_data))             
        print 'Train loss ', epoch_train_loss 


        #Validation
        val_loss = 0
        targets_all = []
        preds_all = []
        nv_batches = int(math.ceil(len(val_data)/float(batch_size)))
        for b in range(nv_batches):
            #Get batch
            val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
            images, targets = process_batch(val_data_batch)
            targets_all += targets
            #
            preds, loss = sess.run([output_layer, loss_node], feed_dict={image_inputs : np.array(images), gender_targets : np.array(targets)})
            preds_all += list(np.argmax(preds,1))
            #Loss
            batch_loss = np.sum(loss)
            val_loss = val_loss + batch_loss 
        #Print val loss
        val_loss = val_loss/float(len(val_data)) 

        print 'Val loss ', val_loss         
        print("Val accuracy ", accuracy_score(targets_all, preds_all))
        print("Val precision ", precision_score(targets_all, preds_all, average = 'macro'))
        print("Val recall ", recall_score(targets_all, preds_all, average = 'macro'))
        print("Val f-score ", f1_score(targets_all, preds_all, average = 'macro'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #Save trained model
            saver.save(sess, model_dump_dir+'model_' + str(e)) 


    #Test
    test_loss = 0
    preds_all = []
    targets_all = []
    nt_batches = int(math.ceil(len(test_data)/float(batch_size)))
    for b in range(nt_batches):
        #Get batch
        test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
        images, targets = process_batch(test_data_batch)
        targets_all += targets
        #Loss
        preds, loss = sess.run([output_layer, loss_node], feed_dict={image_inputs : np.array(images), gender_targets : np.array(targets)})
        preds_all += list(np.argmax(preds,1))
        #Loss
        batch_loss = np.sum(loss)
        test_loss = test_loss + batch_loss 
    #Print val loss
    test_loss = test_loss/float(len(val_data)) 

    print 'Testloss ', test_loss 
    #print(targets_all, preds_all)
    print("Test accuracy ", accuracy_score(targets_all, preds_all))
    print("Test precision ", precision_score(targets_all, preds_all, average = 'macro'))
    print("Test recall ", recall_score(targets_all, preds_all, average = 'macro'))
    print("Test f-score ", f1_score(targets_all, preds_all, average = 'macro'))





    
    
