import numpy as np
from parameters import *
import pickle
from annoy import AnnoyIndex

if use_images == True:
    #Load annoy file for image representations
    url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))
    #print(type(url_to_index))
    #print(url_to_index)
    a = AnnoyIndex(image_vgg_size)
    a.load(annoy_dir+'annoy.ann') 
    #print(a.get_n_items())
    #print(a.get_item_vector(0), a.get_item_vector(1), a.get_item_vector(2))
    print("annoy file loaded")


if use_kg == True:
    kg = pickle.load(open(data_dump_dir+"image_kg.pkl", 'rb'))
    embeddings_google = pickle.load(open(data_dump_dir+'kg_embeddings.pkl', 'rb'))


def image_rep(image_url):
    if image_url in ["", 'RANDOM']:
        v = [0 for e in range(image_size)]
        return np.array(v)

    v = np.array([0 for e in range(image_vgg_size)])
    try:
        index = url_to_index[image_url.strip()]
        v = np.array(a.get_item_vector(index))
    except:      
        if use_images == True:     
            print(image_url + " exception loading from annoy")

    #Append KG info here
    try:
        image_kg = kg[image_url]

        for index in [-3, -1]:
            feature = image_kg[index].lower()
            if feature.strip() == "":
                v = np.hstack((v, np.array([0 for e in range(word_embedding_size)])))
                continue

            features = feature.strip().split()
            if len(features) != 1:
                feature = features[0]

            try:
                p = np.array(embeddings_google[feature])
            except:
                p = np.array([0 for i in range(word_embedding_size)])
                try:
                    print(feature + " not in vocab")
                except:
                    print("can't print feature not in vocab")                
            v = np.hstack((v, p))

        gender = image_kg[3]
        if gender == "men":
            v = np.hstack((v, np.array([1, 0, 0, 0])))
        elif gender == "women":
            v = np.hstack((v, np.array([0, 1, 0, 0])))
        elif gender == "kids":
            v = np.hstack((v, np.array([0, 0, 1, 0])))
        elif gender == "all":
            v = np.hstack((v, np.array([0, 0, 0, 1])))
        else:
            print("Unknown gender ", gender)
            v = np.hstack((v, np.array([0, 0, 0, 0])))

    except:
        v = np.hstack((v, np.array([0 for i in range(image_kg_size)])))
    return v









def get_feed_dict(model, batch_text, batch_image, batch_target_pos, batch_target_negs):
    feed_dict = {}
    
    #print(np.array(batch_text).shape, np.array(batch_image).shape, np.array(batch_target_pos).shape, np.array(batch_target_negs).shape)
    
    #Fill context
    batch_text_ph = [[] for i in range(max_context_len)]
    for i in range(batch_size):
        for j in range(max_context_len):  
            batch_text_ph[j].append(batch_text[i][j]) 
    for i in range(max_context_len):
        batch_text_ph[i] = np.array(batch_text_ph[i]) 
    for ph_i, input_i in zip(model.encoder_text_inputs, batch_text_ph):
        feed_dict[ph_i] = input_i 
         
    #Fill image context
    batch_image_ph = [[] for i in range(max_context_len)]
    for i in range(batch_size):  
        for j in range(max_context_len):
            batch_image_ph[j].append(image_rep(batch_image[i][j]))
    for i in range(max_context_len):
        batch_image_ph[i] = np.array(batch_image_ph[i])
    for ph_i, input_i in zip(model.encoder_image_inputs, batch_image_ph):
        feed_dict[ph_i] = input_i  
    
    #Fill positive image
    batch_target_pos = [[image_rep(image) for image in image_set] for image_set in batch_target_pos]
    feed_dict[model.pos_images] = batch_target_pos

    #Fill negative images
    batch_target_negs = [[image_rep(image) for image in image_set] for image_set in batch_target_negs]
    feed_dict[model.negs_images] = batch_target_negs
    
    #print(np.array(batch_text_ph).shape, np.array(batch_image_ph).shape, np.array(batch_target_pos).shape, np.array(batch_target_negs).shape)
    
    return feed_dict


    
def pad_for_batch_size(batch_text, batch_image, batch_target_pos, batch_target_negs):

    if(len(batch_text) != batch_size):
        pad_size = batch_size - len(batch_target_pos)%batch_size
        #Pad text context
        empty_data = [start_word_id, end_word_id]+[pad_word_id]*(max_utter_len-2)
        empty_data = [empty_data]*max_dialogue_len
        empty_data_mat = [empty_data]*pad_size
        empty_data_mat = np.array(empty_data_mat)
        batch_text = np.vstack((batch_text, empty_data_mat))
        #
        #Pad image context
        empty_data = ""
        empty_data = [empty_data]*max_dialogue_len
        empty_data_mat = [empty_data]*pad_size
        empty_data_mat = np.array(empty_data_mat)
        batch_image = np.vstack((batch_image, empty_data_mat))
        #
        #Pad positive image
        empty_data = ["RANDOM"]
        empty_data_mat = [empty_data]*pad_size
        empty_data_mat = np.array(empty_data_mat)
        batch_target_pos = np.vstack((batch_target_pos, empty_data_mat))
        #
        #Pad negative images
        empty_data = ["RANDOM"] * num_neg_images
        empty_data_mat = [empty_data]*pad_size
        empty_data_mat = np.array(empty_data_mat)
        batch_target_negs = np.vstack((batch_target_negs, empty_data_mat))

    
    return batch_text, batch_image, batch_target_pos, batch_target_negs




def process_batch(model, data_batch):
    #get batch_text, batch_image, batch_target from data_batch
    #data_batch is a batch_size sized list of zips(batch_text, batch_image, batch_target_pos, batch_target_negs)
    batch_text = []
    batch_image = []
    batch_target_pos = []
    batch_target_negs = []

    for instance in data_batch:
        batch_text.append(instance[0])
        batch_image.append(instance[1])
        batch_target_pos.append(instance[2])
        batch_target_negs.append(instance[3])
    batch_text = np.array(batch_text)
    batch_image = np.array(batch_image)
    batch_target_pos = np.array(batch_target_pos) 
    batch_target_negs = np.array(batch_target_negs)
    #batch_text is of batch_size * max_context_len * max_utter_len
    #batch_image is of dimension batch_size * max_context_len 
    #batch_target_pos is of dimension batch_size * 1
    #batch_target_negs is of dimension batch_size * num_neg_images
    
    batch_text, batch_image, batch_target_pos, batch_target_negs = pad_for_batch_size(batch_text, batch_image, batch_target_pos, batch_target_negs)
    
    feed_dict = get_feed_dict(model, batch_text, batch_image, batch_target_pos, batch_target_negs)    
    #batch_text is of batch_size * max_context_len * max_utter_len
    #batch_image is of dimension batch_size * max_context_len * image_size
    #batch_target_pos is of dimension batch_size * 1 * image_size
    #batch_target_negs is of dimension batch_size * num_neg_images * image_size

    return feed_dict



def get_ranks(pos_sim, negs_sim):
    ranks = []
    #print("pos_sim negs_sim shapes ", pos_sim.shape, negs_sim.shape)
    for pos, negs in zip(pos_sim, negs_sim):
        rank = num_neg_images+1
        for neg in negs:
            if pos >= neg:
                rank -= 1
        ranks.append(rank)
    return ranks





def recall(ranks, m):
    correct = 0
    total = 0

    for rank in ranks:
        if rank <= m:
            correct += 1
        total += 1
    return float(correct)/float(total)

