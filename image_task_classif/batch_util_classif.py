import numpy as np
from parameters_classif import *
import pickle
from annoy import AnnoyIndex

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






def pad_for_batch_size(batch_images, batch_gender_targets, batch_color_targets, batch_mat_targets):

    if(len(batch_images) != batch_size):
        pad_size = batch_size - len(batch_images)%batch_size
        
        empty_data_mat = ["RANDOM" for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_images = np.vstack((batch_images, empty_data_mat))

        empty_data_mat = [num_gender_classes for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_gender_targets = np.vstack((batch_gender_targets, empty_data_mat))

        empty_data_mat = [0 for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_color_targets = np.vstack((batch_color_targets, empty_data_mat))

        empty_data_mat = [0 for i in range(pad_size)]
        empty_data_mat = np.array(empty_data_mat)
        batch_mat_targets = np.vstack((batch_mat_targets, empty_data_mat))
    
    return batch_images, batch_gender_targets, batch_color_targets, batch_mat_targets




def process_batch(data_batch):
    
    batch_images = []
    batch_gender_targets_list = []
    batch_color_targets_list = []
    batch_mat_targets_list = []

    for instance in data_batch:
        batch_images.append(instance[0])
        batch_gender_targets_list.append(instance[1])
        batch_color_targets_list.append(instance[2])
        batch_mat_targets_list.append(instance[3])

    batch_images, batch_gender_targets_list, batch_color_targets_list, batch_mat_targets_list = pad_for_batch_size(batch_images, batch_gender_targets_list, batch_color_targets_list, batch_mat_targets_list)
    
    batch_images = [image_rep(image) for image in batch_images]

    return batch_images, batch_gender_targets_list, batch_color_targets_list, batch_mat_targets_list


