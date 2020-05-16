from parameters import *
from HREI import *
from batch_util import *
import pickle, math, random


#Load train and validation data
train_data = pickle.load(open(data_dump_dir+"train_binarized_data.pkl", 'rb'))[:20]
val_data = pickle.load(open(data_dump_dir+"val_binarized_data.pkl", 'rb'))[:5]
print("data loaded")
#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_word_to_id.pkl","rb"))
vocab_size = len(vocab)
#Load number of batches
n_batches = int(math.ceil(len(train_data)/float(batch_size)))



#Create graph
graph1 = tf.Graph()
with graph1.as_default():

    model = HREI(vocab_size)   
    model.create_placeholder()

    model.out, model.code = model.hierarchical_encoder()
    
    model.losses, model.losses_out, model.act_losses_out, model.context_state_out, model.pos_images_out, model.encoded_pos_images_out, model.cosine_sim_pos_out, model.negs_images_reshaped_out, model.encoded_negs_images_out, model.cosine_sim_negs_out = model.loss(model.code)
    model.train_op, _ = model.train(model.losses)
     
    model.simout = model.sim(model.code)   
    
    model.saver = tf.train.Saver()

print("graph made") 


best_val_loss = float("inf")
#Run training
with tf.Session(graph=graph1) as sess:

    sess.run(tf.global_variables_initializer())
    
    if restore_trained == True:
        model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
        print("model loaded")
        
    for e in range(1, epochs+1):
        random.shuffle(train_data)
        train_loss=0
        step = 0
        train_ranks_all = []

        for b in range(n_batches):
            step += 1
            print 'Step ', step   

            if b == n_batches-1:
                model.saver.save(sess, model_dump_dir+'model_' + str(e))          

            #Get batch
            train_data_batch=train_data[b*batch_size:(b+1)*batch_size]
            feed_dict = process_batch(model, train_data_batch)
            #print("FD", feed_dict)
            loss, losses, act_losses, context_state, pos_images, encoded_pos_images, cosine_sim_pos, negs_images_reshaped, encoded_negs_images, cosine_sim_negs, (train_pos_sim, train_negs_sim), _ = sess.run([model.losses, model.losses_out, model.act_losses_out, model.context_state_out, model.pos_images_out, model.encoded_pos_images_out, model.cosine_sim_pos_out, model.negs_images_reshaped_out, model.encoded_negs_images_out, model.cosine_sim_negs_out, model.simout, model.train_op], feed_dict=feed_dict)
            print("LOSS", loss)
            print("LOSSES", losses)
            print("ACT_LOSSES", act_losses)
            print("STATE", context_state)
            print("POS", pos_images)
            print("POS ENC", encoded_pos_images)
            print("POSSIM", cosine_sim_pos)
            print("NEG", negs_images_reshaped)
            print("NEG ENC", encoded_negs_images)
            print("NEGSIM", cosine_sim_negs)
            #Loss
            batch_loss = np.sum(loss)
            per_loss = batch_loss/float(batch_size)    
            print('Epoch  %d Batch %d (Step %d) train loss (avg over batch) =%.6f' %(e, b, step, per_loss))
            train_loss = train_loss + batch_loss
            avg_train_loss = float(train_loss)/float(b+1) 
            #Rank
            train_ranks_all += get_ranks(train_pos_sim, train_negs_sim)       
            # print("Intermediate Ranks", train_ranks_all) 
            # for m in m_for_recall:
            #     print 'Intermediate Train recall @ ', m, ' : ', recall(train_ranks_all, m)


        #Print train loss after each epoch
        print('Epoch %d completed' %(e))
        epoch_train_loss = train_loss/float(len(train_data))             
        print 'Train loss ', epoch_train_loss
        #print(train_ranks_all) 
        for m in m_for_recall:
            print 'Train recall @ ', m, ' : ', recall(train_ranks_all, m)

        #Validation
        val_loss = 0
        val_ranks_all = []
        nv_batches = int(math.ceil(len(val_data)/float(batch_size)))
        for b in range(nv_batches):
            #Get batch
            val_data_batch=val_data[b*batch_size:(b+1)*batch_size]
            feed_dict = process_batch(model, val_data_batch)
            #
            loss, (val_pos_sim, val_negs_sim) = sess.run([model.losses, model.simout], feed_dict=feed_dict)
            #print(loss, losses, train_pos_sim, train_negs_sim)
            #Loss
            batch_loss = np.sum(loss)
            val_loss = val_loss + batch_loss 
            #Ranks
            val_ranks_all += get_ranks(val_pos_sim, val_negs_sim)
        #Print val loss
        val_loss = val_loss/float(len(val_data))           
        print 'Val loss ', val_loss 
        print(val_ranks_all)
        for m in m_for_recall:
            print 'Val recall @ ', m, ' : ', recall(val_ranks_all, m)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #Save trained model
            model.saver.save(sess, model_dump_dir+'model_' + str(e)) 
