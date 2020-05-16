from parameters import *
from HREI import *
from batch_util import *
import pickle, math, random


#Load test data
test_data = pickle.load(open(data_dump_dir+"test_binarized_data.pkl", 'rb'))[:10000]
print("Loaded test data")
#Load vocabulary
vocab = pickle.load(open(data_dump_dir+"vocab_id_to_word.pkl","rb"))
vocab_size = len(vocab)
#Load number of batches
n_batches = int(math.ceil(len(test_data)/float(batch_size)))


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


#Run test 
step = 0   
test_loss = 0
test_ranks_all = []
with tf.Session(graph=graph1) as sess:

    #Restore pre-trained model
    model.saver.restore(sess, tf.train.latest_checkpoint(model_dump_dir))
    print("Loaded model")

    for b in range(n_batches):
        step += 1
        print 'Step ', step

        #Get batch
        test_data_batch=test_data[b*batch_size:(b+1)*batch_size]
        feed_dict = process_batch(model, test_data_batch)

        loss, losses, act_losses, context_state, pos_images, encoded_pos_images, cosine_sim_pos, negs_images_reshaped, encoded_negs_images, cosine_sim_negs, (test_pos_sim, test_negs_sim) = sess.run([model.losses, model.losses_out, model.act_losses_out, model.context_state_out, model.pos_images_out, model.encoded_pos_images_out, model.cosine_sim_pos_out, model.negs_images_reshaped_out, model.encoded_negs_images_out, model.cosine_sim_negs_out, model.simout], feed_dict=feed_dict)
        # print("LOSSES", losses)
        # print("ACT_LOSSES", act_losses)
        # print("STATE", context_state)
        # print("POS", pos_image_reshaped)
        # print("POSSIM", cosine_sim_pos)
        # print("NEG", negs_image_reshaped)
        # print("NEGSIM", cosine_sim_negs)
        #Loss
        batch_loss = np.sum(loss)
        test_loss = test_loss + batch_loss    
        #Ranks
        test_ranks_all += get_ranks(test_pos_sim, test_negs_sim)        
        # for m in m_for_recall:
        #     print 'Intermediate Test recall@ ', str(m), " : ", recall(test_ranks_all, m)   

    #Print test loss
    test_loss = test_loss/float(len(test_data))             
    print 'Test loss ', test_loss  
    for m in m_for_recall:
        print 'Test recall@ ', str(m), " : ", recall(test_ranks_all, m)   