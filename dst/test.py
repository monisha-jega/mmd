import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_dst import *
from parameters_dst import *
import pickle, math
import numpy as np

#Load data
test_data = pickle.load(open(data_dump_dir + "test_dst.pkl"))

#Calculate no of batches
nt_batches = int(math.ceil(len(test_data)/float(batch_size)))

with tf.Session() as sess:

    saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
    print("model loaded")

	#Test
    test_loss = 0
    preds_all = []
    targets_all = []    
    for b in range(nt_batches):
        print("Test batch", b)
        #Get batch
        test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
        test_batch, targets_batch = zip(*test_data_batch)
        targets_all += targets_batch
        #
        #Loss
        preds, loss = sess.run([output_layer, loss_node], feed_dict={test_inputs : np.array(test_batch),
                                                                     targets : np.array(targets_batch)
                                                                    })
        preds_all += list(np.argmax(preds,1))
        #
        #Loss
        batch_loss = np.sum(loss)
        test_loss += batch_loss 

    #Print test loss
    test_loss = test_loss/float(len(val_data)) 
    print 'Test Loss ', test_loss 
    #print(targets_all, preds_all)
    print("Test accuracy ", accuracy_score(targets_all, preds_all))
    print("Test precision ", precision_score(targets_all, preds_all, average = 'micro'))
    print("Test recall ", recall_score(targets_all, preds_all, average = 'micro'))
    print("Test f-score ", f1_score(targets_all, preds_all, average = 'micro'))







    
    
