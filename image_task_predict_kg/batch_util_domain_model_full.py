import numpy as np
from parameters_domain_model_full import *
import pickle
from annoy import AnnoyIndex



kg_embeddings = pickle.load(open(data_dump_dir + 'kg_embeddings.pkl'))

if use_images == True:
    #Load annoy file for image representations
    url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))
    #print(type(url_to_index))
    #print(url_to_index)
    a = AnnoyIndex(image_size)
    a.load(annoy_dir+'annoy.ann') 
    #print(a.get_n_items())
    #print(a.get_item_vector(0), a.get_item_vector(1), a.get_item_vector(2))
    print("annoy file loaded")



def image_rep(image_url):
    v = np.array([0 for e in range(image_size)])
    
    if image_url in ["", 'RANDOM']:        
        return np.array(v)

    try:
        index = url_to_index[image_url.strip()]
        v = np.array(a.get_item_vector(index))
    except:      
        if use_images == True:     
            print(image_url + " exception loading from annoy")

    return v








def pad_for_batch_size(batch_images, batch_gender_targets, batch_mat_targets, batch_color_targets, batch_mat_targets_words, batch_color_targets_words):

    if(len(batch_images) != batch_size):
        pad_size = batch_size - len(batch_images)%batch_size
        
        empty_data_mat = ["RANDOM" for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_images = np.vstack((batch_images, empty_data_mat))

        empty_data_mat = [3 for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_gender_targets = np.vstack((batch_gender_targets, empty_data_mat))

        empty_data_mat = [[0 for i in range(word_embedding_size)] for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_mat_targets = np.vstack((batch_mat_targets, empty_data_mat))

        empty_data_mat = [[0 for i in range(word_embedding_size)] for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_color_targets = np.vstack((batch_color_targets, empty_data_mat))

        empty_data_mat = ["" for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_color_targets = np.vstack((batch_color_targets_words, empty_data_mat))

        empty_data_mat = ["" for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_color_targets = np.vstack((batch_color_targets_words, empty_data_mat))

    
    return batch_images, batch_gender_targets, batch_mat_targets, batch_color_targets, batch_mat_targets_words, batch_color_targets_words




def process_batch(data_batch):
    
    batch_images = []
    batch_gender_targets_list = []
    batch_mat_targets_list = []
    batch_color_targets_list = []
    batch_mat_targets_words = []
    batch_color_targets_words = []

    for instance in data_batch:
        batch_images.append(image_rep(instance[0]))
        batch_gender_targets_list.append(instance[1])
        batch_mat_targets_list.append(kg_embeddings[instance[2]])
        batch_color_targets_list.append(kg_embeddings[instance[3]])

        batch_mat_targets_words.append(instance[2])
        batch_color_targets_words.append(instance[3])

    batch_images, batch_gender_targets_list, batch_mat_targets_list, batch_color_targets_list, batch_mat_targets_words, batch_color_targets_words\
     = pad_for_batch_size(batch_images, batch_gender_targets_list, batch_mat_targets_list, batch_color_targets_list, batch_mat_targets_words, batch_color_targets_words)
    
    return batch_images, batch_gender_targets_list, batch_mat_targets_list, batch_color_targets_list, batch_mat_targets_words, batch_color_targets_words



def similarity(v1, v2):    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.dot(v1, v2)/(norm_v1*norm_v2)


def embedding_accuracy(target_words, pred_embeddings):

    correct_count = 0
    total_count = 0

    for word, pred_embedding in zip(target_words, pred_embeddings):
        target_embedding = kg_embeddings[word] 
        print((target_embedding, pred_embedding))
        sim = similarity(target_embedding, pred_embedding)
        print(sim)
        if sim >= 0.5:
            correct_count += 1
        total_count += 1

    return float(correct_count)/total_count