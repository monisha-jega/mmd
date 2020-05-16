import numpy as np
from parameters import *
import pickle
from annoy import AnnoyIndex

if use_images == True:
    #Load annoy file for image representations
    url_to_index = pickle.load(open(annoy_dir+"ImageUrlToIndex.pkl", 'rb'))
    a = AnnoyIndex(image_size)
    a.load(annoy_dir+'annoy.ann') 
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


def get_reps(images):
    return [image_rep(image) for image in images]
