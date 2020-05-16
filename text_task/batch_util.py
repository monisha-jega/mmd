import numpy as np
from parameters import *
import pickle
from annoy import AnnoyIndex

#If use_images is not true, then annoy file will not be loaded, hence all images will be zeros
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



#Get VGG representation from image URL
def image_rep(image_url):
	v = np.array([0]*image_size)
	if image_url in ["", 'RANDOM']:
		return v
	try:
		index = url_to_index[image_url.strip()]
		v = np.array(a.get_item_vector(index))
	except:      
		if use_images == True:   
			#Print this if image is not found in annoy file  
			print(image_url + " not found in annoy")
	return v


#Convert data into feed_dict format required
def get_feed_dict(model, batch_text, batch_image, batch_target, batch_decoder_input):
    feed_dict = {}

    #Fill context
    batch_text_ph = [[] for i in range(max_context_len)]
    for i in range(batch_size):
        for j in range(max_context_len):  
            batch_text_ph[j].append(batch_text[i][j]) 
    for i in range(max_context_len):
        batch_text_ph[i] = np.array(batch_text_ph[i]) 
    for ph_i, input_i in zip(model.encoder_text_inputs, batch_text_ph):
        feed_dict[ph_i] = input_i  
    #    
    #Fill image context
    batch_image_ph = [[] for i in range(max_context_len)]
    for i in range(batch_size):  
        for j in range(max_context_len):  
            batch_image_ph[j].append(image_rep(batch_image[i][j]))
    for i in range(max_context_len):
        batch_image_ph[i] = np.array(batch_image_ph[i])
    for ph_i, input_i in zip(model.encoder_image_inputs, batch_image_ph):
        feed_dict[ph_i] = input_i  
    #
    #Fill targets
    feed_dict[model.targets] = batch_target
    #
    #Fill decoder inputs
    feed_dict[model.decoder_inputs] = batch_decoder_input

    return feed_dict


    
def pad_for_batch_size(batch_text, batch_image, batch_target): 
    if(len(batch_text) != batch_size):
        pad_size = batch_size - len(batch_target)%batch_size

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
        #Pad target text
        empty_data = [start_word_id, end_word_id]+[pad_word_id]*(max_utter_len-2)
        empty_data_mat = [empty_data]*pad_size
        empty_data_mat = np.array(empty_data_mat)
        batch_target = np.vstack((batch_target, empty_data_mat))
    
    return batch_text, batch_image, batch_target




def process_batch(model, data_batch):
    #data_batch is a batch_size sized list of (batch_text, batch_image, batch_target)
    #get batch_text, batch_image, batch_target from data_batch   
    batch_text, batch_image, batch_target = zip(*data_batch)    
    batch_target_weights = [[0 if word_id == pad_word_id else 1 for word_id in sentence] for sentence in batch_target]
    batch_text = np.array(batch_text)
    batch_image = np.array(batch_image)
    batch_target = np.array(batch_target)        
    #batch_text is of batch_size * max_context_len * max_utter_len
    #batch_image is of dimension batch_size * max_context_len 
    #batch_target is of dimension batch_size * max_utter_len
    
    batch_text, batch_image, batch_target = pad_for_batch_size(batch_text, batch_image, batch_target)
    
    #Do some shifting around
    #Column of pad symbols
    pad_symbols_col = np.reshape(np.array([pad_word_id]*batch_size),(batch_size, 1))
    #Decoder input should start with start symbol but not have end symbol. So end is replaced with pad
    batch_decoder_input = batch_target[:,:-1]
    batch_decoder_input = np.hstack((batch_decoder_input, pad_symbols_col))
    #batch_decoder_input is of dimension batch_size * max_utter_len
    #batch_target should end with end symbol but not have start symbol. So start is removed and pad is added at end.
    batch_target = batch_target[:,1:]
    batch_target = np.hstack((batch_target, pad_symbols_col))
    #batch_target is of dimension batch_size * max_utter_len
    
    feed_dict = get_feed_dict(model, batch_text, batch_image, batch_target, batch_decoder_input)    
    #batch_text is of batch_size * max_context_len * max_utter_len
    #batch_image is of dimension batch_size * max_context_len * image_size
    #batch_target is of dimension batch_size * max_utter_len  
    
    return feed_dict

