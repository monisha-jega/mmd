from parameters_domain_model_full import *
from batch_util_domain_model_full import *
import pickle, math, random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"train_kg.pkl", 'rb'))[:20]
val_data = pickle.load(open(data_dump_dir+"val_kg.pkl", 'rb'))[:5]
test_data = pickle.load(open(data_dump_dir+"test_kg.pkl", 'rb'))[:5]
print("data loaded", len(train_data), len(val_data), len(test_data))
# #Load vocabulary
# vocab = pickle.load(open(data_dump_dir+"vocab_word_to_id.pkl","rb"))
# vocab_size = len(vocab)

#Load number of batches
n_batches = int(math.ceil(len(train_data)/float(batch_size)))


#image_inputs is [batch_size * image_rep_size]
image_inputs_ph = tf.placeholder(tf.float32,[None, image_size], name="image_inputs") 
#gender_targets is [batch_size * 4]
gender_targets_ph = tf.placeholder(tf.int32,[None], name="gender_target") 
#mat_targets is [batch_size * word_embedding_size]
mat_targets_ph = tf.placeholder(tf.float32,[None, word_embedding_size], name="mat_target") 
#color_targets is [batch_size * word_embedding_size]
color_targets_ph = tf.placeholder(tf.float32,[None, word_embedding_size], name="color_target") 

#Input - [batch_size, image_size]
layer1 = tf.layers.dense(image_inputs_ph, hidden_units[0], name="layer1")
layer2 = tf.layers.dense(layer1, hidden_units[1], name="layer2")

gender_output = tf.layers.dense(layer2, gender_dim, name="gender_output")
mat_output = tf.layers.dense(layer2, word_embedding_size, name="mat_output")
color_output = tf.layers.dense(layer2, word_embedding_size, name="color_output")

#labels is [batch_size] (NOT ONE-HOT)
#logits is [batch_size,  gender_dim]
loss_node_gender = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gender_targets_ph, logits=gender_output)
loss_node_mat = tf.losses.mean_squared_error(mat_targets_ph, mat_output)
loss_node_color = tf.losses.mean_squared_error(color_targets_ph, color_output)
loss_total = tf.add(loss_node_gender, tf.add(loss_node_mat, loss_node_color))

optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op=optimizer.minimize(loss_total)

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
            images, gender_targets, mat_targets, color_targets, mat_targets_words, color_targets_words = process_batch(train_data_batch)
            #print(images, targets)
            
            loss, op = sess.run([loss_total, train_op], feed_dict={image_inputs_ph : np.array(images),
                                                                    gender_targets_ph : np.array(gender_targets),
                                                                    mat_targets_ph : np.array(mat_targets),
                                                                    color_targets_ph : np.array(color_targets)
                                                                    })
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
        gender_targets_all = []
        gender_preds_all = []
        mat_preds_all = None
        color_preds_all = None
        mat_targets_words_all = []
        color_targets_words_all = []

        nv_batches = int(math.ceil(len(val_data)/float(batch_size)))
        for b in range(nv_batches):
            #Get batch
            val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
            images, gender_targets, mat_targets, color_targets, mat_targets_words, color_targets_words = process_batch(val_data_batch)
            gender_targets_all += gender_targets
            color_targets_words_all += color_targets_words
            mat_targets_words_all += mat_targets_words
            #
            gender_preds, mat_preds, color_preds, loss = sess.run([gender_output, mat_output, color_output, loss_total], feed_dict={image_inputs_ph : np.array(images),
                                                                                            gender_targets_ph : np.array(gender_targets),
                                                                                            mat_targets_ph : np.array(mat_targets),
                                                                                            color_targets_ph : np.array(color_targets)})
            #print(gender_preds, mat_preds, color_preds)
            #print(images, gender_targets, mat_targets, color_targets, mat_targets_words, color_targets_words)
            gender_preds_all += list(np.argmax(gender_preds,1))
            if b == 0:
                color_preds_all = color_preds
                mat_preds_all = mat_preds
            else:
                color_preds_all = np.vstack((color_preds_all, color_preds))
                mat_preds_all = np.vstack((mat_preds_all, mat_preds))
            #Loss
            batch_loss = np.sum(loss)
            val_loss = val_loss + batch_loss 
        #Print val loss
        val_loss = val_loss/float(len(val_data)) 

        print 'Val loss ', val_loss         
        print("Val A, P, R, F ", accuracy_score(gender_targets_all, gender_preds_all),
                                precision_score(gender_targets_all, gender_preds_all, average = 'macro'),
                                recall_score(gender_targets_all, gender_preds_all, average = 'macro'),
                                f1_score(gender_targets_all, gender_preds_all, average = 'macro'))
        print("Val Color accuracy ", embedding_accuracy(color_targets_words_all, color_preds_all))
        print("Val Mat accuracy ", embedding_accuracy(mat_targets_words_all, mat_preds_all))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #Save trained model
            saver.save(sess, model_dump_dir+'model_' + str(e)) 


    #Test
    test_loss = 0
    gender_targets_all = []
    gender_preds_all = []
    mat_preds_all = None
    color_preds_all = None
    mat_targets_words_all = []
    color_targets_words_all = []

    nt_batches = int(math.ceil(len(test_data)/float(batch_size)))
    for b in range(nt_batches):
        #Get batch
        test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
        images, gender_targets, mat_targets, color_targets, mat_targets_words, color_targets_words = process_batch(test_data_batch)
        gender_targets_all += gender_targets
        color_targets_words_all += color_targets_words
        mat_targets_words_all += mat_targets_words
        #Loss
        gender_preds, mat_preds, color_preds, loss = sess.run([gender_output, mat_output, color_output, loss_total], feed_dict={image_inputs_ph : np.array(images),
                                                                                            gender_targets_ph : np.array(gender_targets),
                                                                                            mat_targets_ph : np.array(mat_targets),
                                                                                            color_targets_ph : np.array(color_targets)
                                                                                            })
        gender_preds_all += list(np.argmax(gender_preds,1))
        if b == 0:
            color_preds_all = color_preds
            mat_preds_all = mat_preds
        else:
            color_preds_all = np.vstack((color_preds_all, color_preds))
            mat_preds_all = np.vstack((mat_preds_all, mat_preds))
        #Loss
        batch_loss = np.sum(loss)
        test_loss = test_loss + batch_loss 
    #Print val loss
    test_loss = test_loss/float(len(test_data)) 

    print 'Testloss ', test_loss 
    #print(targets_all, preds_all)
    print("Test Gender A, P, R, F ", accuracy_score(gender_targets_all, gender_preds_all),
                                precision_score(gender_targets_all, gender_preds_all, average = 'macro'),
                                recall_score(gender_targets_all, gender_preds_all, average = 'macro'),
                                f1_score(gender_targets_all, gender_preds_all, average = 'macro'))
    print("Test Color accuracy ", embedding_accuracy(color_targets_words_all, color_preds_all))
    print("Test Mat accuracy ", embedding_accuracy(mat_targets_words_all, mat_preds_all))




    
    
