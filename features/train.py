from parameters import *
from util import *
import pickle, math, random, os, sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"train.pkl", 'rb'))[:]
val_data = pickle.load(open(data_dump_dir+"val.pkl", 'rb'))[:]
test_data = pickle.load(open(data_dump_dir+"test.pkl", 'rb'))[:]
print("data loaded", len(train_data), len(val_data), len(test_data))


#image_inputs is [batch_size * image_rep_size]
image_inputs = tf.placeholder(tf.float32,[None, image_size], name="image_inputs") 
#targets is [batch_size]
targets = tf.placeholder(tf.int32,[None], name="targets") 

#Input - [batch_size, image_size]
#Output - [batch_size, feature_size]
layer1 = tf.layers.dense(image_inputs, hidden_units[0], name="layer1")
layer2 = tf.layers.dense(layer1, hidden_units[1], name="layer2")
output_layer = tf.layers.dense(layer2, feature_size, name="output_layer")

#labels is [batch_size] (NOT ONE-HOT)
#logits is [batch_size,  4]
loss_node = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits=output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer_gender.minimize(loss_node)

saver = tf.train.Saver()
print("tf graph created")



best_val_loss = float("inf")
#Load number of batches
n_batches = len(train_data)/batch_size
nv_batches = len(val_data)/batch_size
nt_batches = len(test_data)/batch_size

#Run training
with tf.Session() as sess:
    
    if restore_trained == True:
        saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
        print("model loaded")
    else:
        sess.run(tf.global_variables_initializer())
    
    for e in range(1, epochs+1):
        random.shuffle(train_data)

        train_loss=0
        
        for b in range(n_batches):
            print 'Step ', b+1

            #Get batch
            images_batch, targets_batch = zip(*(train_data[b*batch_size : (b+1)*batch_size]))
            images_batch, targets_batch = np.array(get_reps(images_batch)), np.array(targets_batch)
            #Train
            batch_loss, op = sess.run([loss_node, train_op], feed_dict={image_inputs : images_batch, targets : targets_batch})
            #Loss
            batch_loss = np.sum(batch_loss)
            per_loss = batch_loss/float(batch_size)    
            print('Epoch  %d Batch %d train loss (avg over batch) =%.6f' %(e, b, per_loss))
            train_loss = train_loss + batch_loss           
        
        #Print train loss after each epoch
        train_loss = train_loss/len(train_data)
        print('Epoch %d completed with loss %.6f' %(e, train_loss))

        #Validation
        val_loss = 0
        targets_all = []
        preds_all = []

        for b in range(nv_batches):
            #Get batch
            images_batch, targets_batch = zip(*(val_data[b*batch_size : (b+1)*batch_size]))
            images_batch, targets_batch = np.array(get_reps(images_batch)), np.array(targets_batch)
            targets += targets_batch
            #Train
            preds, batch_loss = sess.run([output_layer, loss_node], feed_dict={image_inputs : images_batch, targets : targets_batch})
            preds_all += list(np.argmax(preds,1))
            #Loss
            batch_loss = np.sum(loss)
            val_loss = val_loss + batch_loss         
        
        #Print val loss
        val_loss = val_loss/float(len(val_data)) 
        print 'Val loss ', val_loss         
        print("Val accuracy ", accuracy_score(targets_all, preds_all))
        print("Val precision ", precision_score(targets_all, preds_all, average = 'micro'))
        print("Val recall ", recall_score(targets_all, preds_all, average = 'micro'))
        print("Val f-score ", f1_score(targets_all, preds_all, average = 'micro'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #Save trained model
            saver.save(sess, model_dump_dir+'model_' + str(e)) 


    #Test
    test_loss = 0
    targets_all = []
    preds_all = []

    for b in range(nt_batches):
        #Get batch
        images_batch, targets_batch = zip(*(test_data[b*batch_size : (b+1)*batch_size]))
        images_batch, targets_batch = np.array(get_reps(images_batch)), np.array(targets_batch)
        targets += targets_batch
        #Train
        preds, batch_loss = sess.run([output_layer, loss_node], feed_dict={image_inputs : images_batch, targets : targets_batch})
        preds_all += list(np.argmax(preds,1))
        #Loss
        batch_loss = np.sum(loss)
        test_loss = test_loss + batch_loss         
    
    #Print val loss
    test_loss = test_loss/float(len(test_data)) 
    print 'Test loss ', test_loss         
    print("Test accuracy ", accuracy_score(targets_all, preds_all))
    print("Test precision ", precision_score(targets_all, preds_all, average = 'micro'))
    print("Test recall ", recall_score(targets_all, preds_all, average = 'micro'))
    print("Test f-score ", f1_score(targets_all, preds_all, average = 'micro'))